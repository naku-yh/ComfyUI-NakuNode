import { app } from "../../scripts/app.js";

// --- 配置常量 ---
const CANVAS_SIZE = 320;
const CENTER_X = 160;
const CENTER_Y = 160;
const RADIUS_WIDE = 140;
const RADIUS_MEDIUM = 90;
const RADIUS_CLOSE = 50;

// 颜色
const COLOR_BG = "#1a1a1a";
const COLOR_GRID_LINES = "#444";
const COLOR_TEXT = "#888";
const COLOR_ACTIVE = "#ffbd45";
const COLOR_HIGHLIGHT = "#ffffff";

// 数据
const ELEVATION_STEPS = [-30, 0, 30, 60];
const DISTANCE_MAP = {
    "close-up": RADIUS_CLOSE,
    "medium shot": RADIUS_MEDIUM,
    "wide shot": RADIUS_WIDE
};
const DISTANCE_REVERSE_MAP = {
    [RADIUS_CLOSE]: "close-up",
    [RADIUS_MEDIUM]: "medium shot",
    [RADIUS_WIDE]: "wide shot"
};

// --- 自定义 Widget 类 ---
class NakuVNCCS_CameraWidget {
    constructor(node, inputName, inputData, app) {
        this.node = node;
        this.inputName = inputName;
        this.app = app;

        // 内部状态
        this.state = {
            azimuth: 0,
            elevation: 0,
            distance: "medium shot",
            include_trigger: true
        };

        // 尝试加载初始状态
        try {
            if (this.node.widgets && this.node.widgets[0]) {
                const loaded = JSON.parse(this.node.widgets[0].value);
                this.state = { ...this.state, ...loaded };
            }
        } catch (e) { }

        this.isDragging = false;
        this.dragMode = null; // 'azimuth' 或 'elevation'

        // 创建 Canvas 元素
        this.canvas = document.createElement("canvas");
        this.canvas.width = CANVAS_SIZE;
        this.canvas.height = CANVAS_SIZE;
        this.canvas.style.borderRadius = "4px";
        this.ctx = this.canvas.getContext("2d");

        // UI 事件监听器
        this.canvas.style.touchAction = "none";
        this.canvas.addEventListener("pointerdown", this.onPointerDown.bind(this));

        // 使用 document 监听 move/up 事件，即使鼠标移出 canvas 也能捕获
        this.canvas.addEventListener("pointermove", this.onPointerMove.bind(this));
        this.canvas.addEventListener("pointerup", this.onPointerUp.bind(this));
        this.canvas.addEventListener("pointercancel", this.onPointerUp.bind(this));

        // 初始绘制
        this.draw();
    }

    // --- 绘制逻辑 ---
    draw() {
        const ctx = this.ctx;
        ctx.fillStyle = COLOR_BG;
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        this.drawFrontIndicator(ctx); // 先绘制，使其在背景层
        this.drawGrid(ctx);
        this.drawSubject(ctx);
        this.drawCameraTriangle(ctx);
        this.drawElevationBar(ctx);
        this.drawInfoText(ctx);
    }

    drawFrontIndicator(ctx) {
        // 绘制从底部指向中心的箭头，表示正面（FRONT）
        ctx.save();
        ctx.translate(CENTER_X, CENTER_Y);

        // 文字 "正面"
        ctx.fillStyle = "rgba(255, 255, 255, 0.3)"; // 半透明白色
        ctx.font = "bold 16px sans-serif";
        ctx.textAlign = "center";

        // 将文字放在圆圈内，避免被裁剪
        ctx.fillText("正面", 0, RADIUS_WIDE - 40);

        // 从底部向内的箭头
        ctx.beginPath();
        // 轴：从边缘附近 (135) 到内部 (115)
        ctx.moveTo(0, RADIUS_WIDE - 5);
        ctx.lineTo(0, RADIUS_WIDE - 25);

        // 箭头
        ctx.moveTo(0, RADIUS_WIDE - 25);
        ctx.lineTo(-5, RADIUS_WIDE - 18);
        ctx.moveTo(0, RADIUS_WIDE - 25);
        ctx.lineTo(5, RADIUS_WIDE - 18);

        ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.restore();
    }

    drawGrid(ctx) {
        // 绘制圆圈
        ctx.strokeStyle = COLOR_GRID_LINES;
        ctx.lineWidth = 1;

        [RADIUS_CLOSE, RADIUS_MEDIUM, RADIUS_WIDE].forEach(r => {
            ctx.beginPath();
            ctx.arc(CENTER_X, CENTER_Y, r, 0, Math.PI * 2);
            ctx.stroke();
        });

        // 绘制轴线（X 形表示 45 度角）
        ctx.beginPath();
        ctx.moveTo(CENTER_X - RADIUS_WIDE, CENTER_Y);
        ctx.lineTo(CENTER_X + RADIUS_WIDE, CENTER_Y);
        ctx.moveTo(CENTER_X, CENTER_Y - RADIUS_WIDE);
        ctx.lineTo(CENTER_X, CENTER_Y + RADIUS_WIDE);

        // 对角线
        const diag = RADIUS_WIDE * 0.707;
        ctx.moveTo(CENTER_X - diag, CENTER_Y - diag);
        ctx.lineTo(CENTER_X + diag, CENTER_Y + diag);
        ctx.moveTo(CENTER_X + diag, CENTER_Y - diag);
        ctx.lineTo(CENTER_X - diag, CENTER_Y + diag);
        ctx.stroke();
    }

    drawSubject(ctx) {
        // 中心的方框
        ctx.fillStyle = "#666";
        ctx.fillRect(CENTER_X - 6, CENTER_Y - 6, 12, 12);
    }

    drawCameraTriangle(ctx) {
        const r = DISTANCE_MAP[this.state.distance];
        // 将方位角转换为数学角度
        // 0 度 = 正面（底部，PI/2）
        // 90 度 = 右侧（0）
        // 180 度 = 背面（顶部，-PI/2）
        // 270 度 = 左侧（PI）

        // 公式：Angle = PI/2 - (Azimuth * PI/180)
        const angleRad = (Math.PI / 2) - (this.state.azimuth * (Math.PI / 180));

        const cx = CENTER_X + r * Math.cos(angleRad);
        const cy = CENTER_Y + r * Math.sin(angleRad);

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angleRad + Math.PI / 2); // 指向中心

        // 三角形形状
        ctx.fillStyle = COLOR_ACTIVE;
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 10); // 指向内部
        ctx.lineTo(-8, -8);
        ctx.lineTo(8, -8);
        ctx.closePath();
        ctx.fill();
        ctx.stroke(); // 添加轮廓以提高可见性

        ctx.restore();
    }

    drawElevationBar(ctx) {
        // 右侧的简单垂直滑块
        const barX = CANVAS_SIZE - 20;
        const barH = 200;
        const barY = (CANVAS_SIZE - barH) / 2;

        // 轨道线
        ctx.strokeStyle = COLOR_GRID_LINES;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(barX, barY);
        ctx.lineTo(barX, barY + barH);
        ctx.stroke();

        // 刻度
        // -30（底部）到 60（顶部）
        ELEVATION_STEPS.forEach(step => {
            const norm = (step + 30) / 90; // 0..1
            const y = barY + barH - (norm * barH);

            ctx.fillStyle = (step === this.state.elevation) ? COLOR_ACTIVE : "#666";
            ctx.beginPath();
            ctx.arc(barX, y, 4, 0, Math.PI * 2);
            ctx.fill();

            // 文字标签
            if (Math.abs(step - this.state.elevation) < 0.1 || step % 30 === 0) {
                ctx.fillStyle = "#888";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "right";
                ctx.fillText(step + "°", barX - 8, y + 3);
            }
        });

        // 当前指示器手柄
        const currentNorm = (this.state.elevation + 30) / 90;
        const curY = barY + barH - (currentNorm * barH);
        ctx.fillStyle = COLOR_ACTIVE;
        ctx.beginPath();
        ctx.arc(barX, curY, 6, 0, Math.PI * 2);
        ctx.fill();
    }

    drawInfoText(ctx) {
        ctx.fillStyle = COLOR_TEXT;
        ctx.font = "12px monospace";
        ctx.textAlign = "left";
        
        // 距离映射（英文转中文）
        const distanceMap = {
            "close-up": "特写",
            "medium shot": "中景",
            "wide shot": "广角"
        };
        const distanceCn = distanceMap[this.state.distance] || this.state.distance;
        
        ctx.fillText(`方位角：${this.state.azimuth}°`, 10, CANVAS_SIZE - 40);
        ctx.fillText(`仰角：${this.state.elevation}°`, 10, CANVAS_SIZE - 25);
        ctx.fillText(`距离：${distanceCn}`, 10, CANVAS_SIZE - 10);

        // 触发词状态指示器
        ctx.fillStyle = this.state.include_trigger ? "#4a4" : "#a44";
        ctx.fillRect(CANVAS_SIZE - 20, CANVAS_SIZE - 20, 10, 10);
    }

    // --- 交互 ---
    onPointerDown(e) {
        this.canvas.setPointerCapture(e.pointerId);
        this.isDragging = true;
        this.handlePointer(e);
    }

    onPointerMove(e) {
        if (!this.isDragging) return;
        this.handlePointer(e);
    }

    onPointerUp(e) {
        this.isDragging = false;
        this.dragMode = null;
        this.canvas.releasePointerCapture(e.pointerId);
    }

    handlePointer(e) {
        const rect = this.canvas.getBoundingClientRect();
        // 计算缩放因子，以防 UI 缩放
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        if (this.dragMode === 'elevation') {
            this.updateElevation(y);
            return;
        }

        if (!this.dragMode) {
            // 检查仰角条
            const barX = CANVAS_SIZE - 20;
            if (Math.abs(x - barX) < 20) {
                this.dragMode = 'elevation';
                this.updateElevation(y);
                return;
            }

            // 检查触发词框
            if (x > CANVAS_SIZE - 30 && y > CANVAS_SIZE - 30) {
                this.state.include_trigger = !this.state.include_trigger;
                this.updateNode();
                this.draw();
                this.isDragging = false;
                return;
            }

            this.dragMode = 'azimuth';
        }

        // 默认：方位角/距离
        this.updatePos(x, y);
    }

    // 逻辑更新
    updatePos(x, y) {
        // 1. 计算角度
        const dx = x - CENTER_X;
        const dy = y - CENTER_Y;

        let angleRad = Math.atan2(dy, dx);
        let deg = (Math.PI / 2 - angleRad) * (180 / Math.PI);

        // 标准化到 0-360
        if (deg < 0) deg += 360;
        if (deg >= 360) deg -= 360;

        // 吸附到 45 度
        this.state.azimuth = Math.round(deg / 45) * 45;
        if (this.state.azimuth >= 360) this.state.azimuth = 0;

        // 2. 计算距离（半径）
        const dist = Math.sqrt(dx * dx + dy * dy);

        // 吸附逻辑：只有当靠近圆环区域时才改变距离
        const activeZone = RADIUS_WIDE + 60; // 200px

        if (dist < activeZone) {
            // 吸附到圆环
            const dists = [RADIUS_CLOSE, RADIUS_MEDIUM, RADIUS_WIDE];
            const closest = dists.reduce((prev, curr) =>
                Math.abs(curr - dist) < Math.abs(prev - dist) ? curr : prev
            );
            this.state.distance = DISTANCE_REVERSE_MAP[closest];
        }

        this.updateNode();
        this.draw();
    }

    updateElevation(y) {
        const barH = 200;
        const barY = (CANVAS_SIZE - barH) / 2;

        // 从 Y 反向映射到角度
        let norm = (barY + barH - y) / barH;
        if (norm < 0) norm = 0;
        if (norm > 1) norm = 1;

        // Deg = norm * 90 - 30
        let deg = norm * 90 - 30;

        // 吸附到步骤 [-30, 0, 30, 60]
        const closest = ELEVATION_STEPS.reduce((prev, curr) =>
            Math.abs(curr - deg) < Math.abs(prev - deg) ? curr : prev
        );

        this.state.elevation = closest;
        this.updateNode();
        this.draw();
    }

    updateNode() {
        // 将状态序列化到隐藏的 widget
        if (this.node.widgets && this.node.widgets[0]) {
            this.node.widgets[0].value = JSON.stringify(this.state);
        }
    }
}


// --- 扩展注册 ---
app.registerExtension({
    name: "NakuNode.镜头可视化控制",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NakuNode_镜头可视化控制") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // 添加自定义 Widget
                const widget = new NakuVNCCS_CameraWidget(this, "camera_camera", {}, app);

                // 将 canvas 添加到节点的 DOM
                this.addDOMWidget("CameraControl", "canvas", widget.canvas, {
                    serialize: false, // 我们不序列化 canvas 本身
                    hideOnZoom: false
                });

                // 强制初始更新以同步隐藏的 widget
                widget.updateNode();

                // 保持合适的尺寸
                this.setSize([340, 380]);
            };
        }
    }
});
