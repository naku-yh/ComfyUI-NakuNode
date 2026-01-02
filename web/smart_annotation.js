import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfy.NAKUSmartAnnotation_NakuNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "NAKUSmartAnnotation_NakuNodes") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.points = [];
            this.imgObj = null;
            this.isMarkingMode = false;
            this.hoverPointIndex = -1;

            // 用于记录当前文件名，以便检测切换
            this.currentFileName = "";

            // --- 1. 按钮组 ---
            this.toggleButton = this.addWidget("button", "开始标记", null, () => {
                this.toggleMarkingMode();
            });

            this.clearButton = this.addWidget("button", "清空所有标记", null, () => {
                if (this.points.length > 0) {
                    if(confirm("确定要清空所有标记吗？")) {
                        this.points = [];
                        this.updateWidget();
                    }
                }
            });

            // 添加颜色选择下拉菜单
            this.colorWidget = this.addWidget("combo", "标注颜色", "红色", () => {}, { values: ["红色", "蓝色", "黄色", "白色", "黑色"] });

            // --- 2. 隐藏数据 Widget ---
            if (!this.widgets) this.widgets = [];
            let pWidget = this.widgets.find(w => w.name === "points_data");
            if (!pWidget) {
                pWidget = this.addWidget("text", "points_data", "[]", () => {}, { serialize: true });
            }
            this.pointsWidget = pWidget;
            pWidget.type = "hidden";
            pWidget.computeSize = () => [0, -4];
            pWidget.draw = function() {};

            if (this.pointsWidget.value) {
                try {
                    const parsed = JSON.parse(this.pointsWidget.value);
                    if (Array.isArray(parsed)) this.points = parsed;
                } catch (e) {}
            }

            // --- 3. 图像加载与自动清空逻辑 ---
            const imageWidget = this.widgets.find(w => w.name === "image");

            this.loadImage = (filename) => {
                if (!filename) return;
                const img = new Image();
                img.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&t=${Date.now()}`);
                img.onload = () => {
                    this.imgObj = img;
                    this.fitNodeSize();
                };
            };

            if (imageWidget) {
                // 初始化时记录当前文件名
                this.currentFileName = imageWidget.value;

                const origCallback = imageWidget.callback;
                imageWidget.callback = (v) => {
                    // 【核心修复】检测图片切换
                    // 如果新文件名 v 和当前记录的不同，说明用户切换了图片 -> 清空标记
                    if (this.currentFileName !== v) {
                        this.points = []; // 清空数组
                        this.updateWidget(); // 更新后端
                        this.currentFileName = v; // 更新记录
                    }

                    if (origCallback) origCallback(v);
                    this.loadImage(v);
                };

                // 初始加载
                if (imageWidget.value) this.loadImage(imageWidget.value);
            }

            // --- 4. 辅助函数 ---
            this.toggleMarkingMode = () => {
                this.isMarkingMode = !this.isMarkingMode;
                const newLabel = this.isMarkingMode ? "完成标记" : "开始标记";
                this.toggleButton.name = newLabel;
                if(this.toggleButton.label !== undefined) this.toggleButton.label = newLabel;
                app.graph.setDirtyCanvas(true, true);
            };

            this.getHeaderHeight = () => {
                let h = 40; // 增加基础高度以容纳提示文字
                if (this.widgets) {
                    for (const w of this.widgets) {
                        if (w.name === "points_data" || w.type === "hidden") continue;
                        let wh = w.last_h || 20;
                        if (w.computeSize) {
                             const size = w.computeSize(this.size[0]);
                             if (size && size[1]) wh = size[1];
                        }
                        h += wh + 10;
                    }
                }
                return h + 10;
            };

            // 核心坐标计算
            this.computeDrawLayout = () => {
                const headerH = this.getHeaderHeight();
                const nodeW = this.size[0];

                let imgH = 200;
                if (this.imgObj) {
                    const aspect = this.imgObj.width / this.imgObj.height;
                    imgH = nodeW / aspect;
                }

                return {
                    headerHeight: headerH,
                    imgX: 0,
                    imgY: headerH,
                    imgW: nodeW,
                    imgH: imgH,
                    totalH: headerH + imgH
                };
            };

            this.fitNodeSize = () => {
                const layout = this.computeDrawLayout();
                if (Math.abs(this.size[1] - layout.totalH) > 5) {
                    this.setSize([this.size[0], layout.totalH]);
                }
                app.graph.setDirtyCanvas(true, true);
            };

            if(this.size[0] < 250) this.setSize([250, 300]);

            return r;
        };

        // --- 5. 缩放锁定 ---
        nodeType.prototype.onResize = function(size) {
            if (this.imgObj) {
                const layout = this.computeDrawLayout();
                size[1] = layout.totalH;
            }
        };

        // --- 6. 绘制背景 ---
        nodeType.prototype.onDrawBackground = function (ctx) {
            if (this.flags.collapsed) return;
            const layout = this.computeDrawLayout();

            ctx.fillStyle = "rgba(30, 30, 30, 1)";
            ctx.fillRect(0, 0, this.size[0], layout.headerHeight);

            if (this.imgObj) {
                ctx.drawImage(this.imgObj, layout.imgX, layout.imgY, layout.imgW, layout.imgH);
            } else {
                ctx.fillStyle = "#222";
                ctx.fillRect(layout.imgX, layout.imgY, layout.imgW, layout.imgH);
                ctx.fillStyle = "#888";
                ctx.textAlign = "center";
                ctx.fillText("请选择图片", layout.imgW / 2, layout.imgY + 40);
            }

            if (this.isMarkingMode) {
                ctx.lineWidth = 4;
                ctx.strokeStyle = "#FF0000";
                ctx.strokeRect(layout.imgX, layout.imgY, layout.imgW, layout.imgH);

                ctx.fillStyle = "#FF4444";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "left";
                // 【提示更新】
                // 在节点头部区域绘制提示文字
                ctx.fillText("⚠ 标记模式: 左键加点 / Shift+左键删点", 10, 25);
            }
        };

        // --- 7. 绘制前景 ---
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (this.flags.collapsed || !this.imgObj) return;
            const layout = this.computeDrawLayout();

            ctx.save();
            ctx.translate(layout.imgX, layout.imgY);

            ctx.font = "bold 27px Arial";  // 缩小30% (38 * 0.7 = 26.6，约等于27)
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            // 定义颜色映射
            const colorMap = {
                "红色": "#FF0000",
                "蓝色": "#0000FF",
                "黄色": "#FFFF00",
                "白色": "#FFFFFF",
                "黑色": "#000000"
            };
            const fillColor = colorMap[this.colorWidget?.value || "红色"] || "#FF0000";

            for (let i = 0; i < this.points.length; i++) {
                const p = this.points[i];
                const cx = p.x * layout.imgW;
                const cy = p.y * layout.imgH;

                ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = 4;

                const radius = 30; // 增大50%（从20到30）
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);

                if (i === this.hoverPointIndex) {
                    ctx.fillStyle = "rgba(255, 255, 0, 0.9)"; // 黄色
                    ctx.strokeStyle = "red";
                } else {
                    ctx.fillStyle = fillColor;   // 使用选择的颜色
                    ctx.strokeStyle = "white";
                }

                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();

                ctx.shadowBlur = 0;
                ctx.fillStyle = (i === this.hoverPointIndex) ? "black" : "white";
                // 在圆内居中文字
                ctx.fillText(i + 1, cx, cy);
            }
            ctx.restore();
        };

        // --- 8. 鼠标移动 ---
        nodeType.prototype.onMouseMove = function(e, local_pos, canvas) {
            if (!this.imgObj || !this.isMarkingMode) {
                this.hoverPointIndex = -1;
                return;
            }

            const layout = this.computeDrawLayout();
            if (local_pos[1] < layout.imgY) {
                this.hoverPointIndex = -1;
                return;
            }

            const mouseX = local_pos[0] - layout.imgX;
            const mouseY = local_pos[1] - layout.imgY;

            let hitIndex = -1;
            const hitRadius = 30;

            for (let i = 0; i < this.points.length; i++) {
                const px = this.points[i].x * layout.imgW;
                const py = this.points[i].y * layout.imgH;
                const dist = Math.sqrt(Math.pow(px - mouseX, 2) + Math.pow(py - mouseY, 2));
                if (dist < hitRadius) {
                    hitIndex = i;
                    break;
                }
            }

            if (this.hoverPointIndex !== hitIndex) {
                this.hoverPointIndex = hitIndex;
                app.graph.setDirtyCanvas(true, true);
            }
        };

        // --- 9. 鼠标点击 (Shift+左键逻辑) ---
        nodeType.prototype.onMouseDown = function (e, local_pos, canvas) {
            if (!this.imgObj || !this.isMarkingMode) return false;

            const layout = this.computeDrawLayout();
            if (local_pos[1] < layout.imgY) return false;

            // 只处理左键 (Button 0)
            if (e.button === 0) {
                // 计算相对坐标
                const relX = (local_pos[0] - layout.imgX) / layout.imgW;
                const relY = (local_pos[1] - layout.imgY) / layout.imgH;

                const clickedIndex = this.hoverPointIndex;

                // 逻辑 A: 按住 Shift -> 删除模式
                if (e.shiftKey) {
                    if (clickedIndex !== -1) {
                        this.points.splice(clickedIndex, 1);
                        this.hoverPointIndex = -1;
                        this.updateWidget();
                    }
                    // 即使没点中，只要按了Shift，也拦截事件，防止拖拽
                    return true;
                }

                // 逻辑 B: 普通点击 -> 新增模式
                // (前提是鼠标没有悬停在现有点上，避免在点上重复添加)
                if (clickedIndex === -1) {
                    this.points.push({ x: relX, y: relY });
                    this.updateWidget();
                }

                return true;
            }

            return false;
        };

        nodeType.prototype.updateWidget = function () {
            if (this.pointsWidget) {
                this.pointsWidget.value = JSON.stringify(this.points);
            }
            app.graph.setDirtyCanvas(true, true);
        };

        // 既然右键恢复默认，这里就不需要强行返回 null 了
        // nodeType.prototype.getExtraMenuOptions = ... (删除此段代码即可)
    }
});