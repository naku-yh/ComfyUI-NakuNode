import { app } from "/scripts/app.js";

// NAKU Annotation Helper V2 Extension
app.registerExtension({
    name: "Comfy.NAKUAnnotationHelperV2_NakuNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NAKUAnnotationHelperV2_NakuNodes") {

            class NAKUAnnotationWidgetV2 {
                constructor(node) {
                    this.node = node;
                    this.drawing = false;
                    this.currentTool = "brush";
                    this.currentColor = "#FF0000";  // 默认红色
                    this.brushSize = 15;

                    this.history = [];
                    this.historyIndex = -1;

                    this.element = document.createElement("div");
                    this.element.className = "naku-annotation-widget-v2";
                    this.node.addDOMWidget("annotation_widget", "custom", this.element);

                    // 获取节点上的控件
                    this.annotationDataWidget = node.widgets.find(w => w.name === "annotation_data");

                    // 隐藏用于数据传输的文本框
                    if (this.annotationDataWidget?.inputEl) {
                        this.annotationDataWidget.inputEl.style.display = "none";
                    }

                    this.setupDOM();
                    this.setupCanvas();
                    this.setupEvents();
                    this.handleConnectionChange();
                }

                setupDOM() {
                    this.toolbar = document.createElement("div");
                    this.toolbar.className = "toolbar";

                    this.toolButtons = {
                        brush: this.createButton("画笔", () => this.setTool("brush")),
                        eraser: this.createButton("橡皮擦", () => this.setTool("eraser")),
                        rect: this.createButton("方框", () => this.setTool("rect")),
                    };

                    this.undoBtn = this.createButton("撤销", () => this.undo());
                    this.redoBtn = this.createButton("重做", () => this.redo());

                    const colorOptions = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFFFFF", "#000000"];
                    this.colorPalette = this.createColorPalette(colorOptions);

                    const sliderLabel = this.createLabel("画笔大小:");
                    const brushSlider = this.createSlider(1, 100, this.brushSize, (e) => { this.brushSize = e.target.value; });

                    this.loadImageButton = this.createButton("载入图像", () => this.loadImage());
                    this.clearBtn = this.createButton("清除", () => this.clearDrawingAndHistory());
                    this.okBtn = this.createButton("确认", () => this.finalize(), "ok-button");

                    this.toolbar.append(
                        this.toolButtons.brush,
                        this.toolButtons.eraser,
                        this.toolButtons.rect,
                        this.undoBtn,
                        this.redoBtn,
                        this.colorPalette,
                        sliderLabel,
                        brushSlider,
                        this.loadImageButton,
                        this.clearBtn,
                        this.okBtn
                    );

                    this.element.appendChild(this.toolbar);
                    this.setTool("brush");
                }

                setupCanvas() {
                    this.canvasContainer = document.createElement("div");
                    this.canvasContainer.className = "canvas-container";

                    this.bgCanvas = document.createElement("canvas");      // 用于显示原始图像
                    this.drawCanvas = document.createElement("canvas");    // 用于显示彩色蒙版和用户的所有绘制
                    this.previewCanvas = document.createElement("canvas"); // 用于实时预览（如画框）

                    [this.bgCanvas, this.drawCanvas, this.previewCanvas].forEach(c => {
                        c.className = "resizing-canvas";
                        this.canvasContainer.appendChild(c);
                    });

                    this.bgCtx = this.bgCanvas.getContext("2d");
                    this.drawCtx = this.drawCanvas.getContext("2d", { willReadFrequently: true });
                    this.previewCtx = this.previewCanvas.getContext("2d");

                    this.element.appendChild(this.canvasContainer);
                }

                setupEvents() {
                    this.previewCanvas.addEventListener("mousedown", (e) => this.startDrawing(e));
                    this.previewCanvas.addEventListener("mousemove", (e) => this.draw(e));
                    this.previewCanvas.addEventListener("mouseup", (e) => this.stopDrawing(e));
                    this.previewCanvas.addEventListener("mouseleave", (e) => this.stopDrawing(e, true));
                }

                async loadImage() {
                    this.clearAll();

                    const imageInput = this.node.inputs.find(inp => inp.name === '图像');

                    try {
                        const originalImage = await this.loadImageFromInput(imageInput, '图像');
                        if (!originalImage) {
                            alert("[NAKU] 无法加载图像。请确保图像输入已连接并已生成预览。");
                            return;
                        }

                        // 根据加载的图像设置画布尺寸
                        this.setCanvasSize(originalImage.naturalWidth, originalImage.naturalHeight);
                        this.bgCtx.drawImage(originalImage, 0, 0);

                        console.log("[NAKU] 图像加载成功。");

                    } catch (error) {
                        this.clearAll();
                        alert(error.message);
                    }
                }

                loadImageFromInput(input, inputNameForError) {
                    return new Promise((resolve, reject) => {
                        if (!input || input.link === null) return resolve(null);

                        const linkInfo = this.node.graph.links[input.link];
                        if (!linkInfo) return reject(new Error(`[NAKU] 找不到 ${inputNameForError} 的连接线信息。`));

                        const upstreamNode = this.node.graph.getNodeById(linkInfo.origin_id);
                        if (!upstreamNode) return reject(new Error(`[NAKU] 找不到 ${inputNameForError} 的上游节点。`));

                        if (upstreamNode.dirty) {
                             return reject(new Error(`[NAKU] 上游节点 (${upstreamNode.title}) 需要重新运行。请点击 "Queue Prompt"。`));
                        }

                        let attempts = 20;
                        const tryLoad = () => {
                            const imgData = upstreamNode.imgs?.[0];
                            if (imgData && imgData.complete && imgData.naturalWidth > 0) {
                                resolve(imgData);
                            } else {
                                attempts--;
                                if (attempts > 0) {
                                    setTimeout(tryLoad, 300);
                                } else {
                                    reject(new Error(`[NAKU] 从 ${inputNameForError} 获取预览图超时。请确保工作流已运行, 或再次点击加载按钮。`));
                                }
                            }
                        };
                        tryLoad();
                    });
                }

                saveState() {
                    if (this.historyIndex < this.history.length - 1) {
                        this.history = this.history.slice(0, this.historyIndex + 1);
                    }
                    this.history.push(this.drawCanvas.toDataURL("image/png"));
                    this.historyIndex = this.history.length - 1;
                    this.updateHistoryButtons();
                }

                restoreState(index) {
                    const img = new Image();
                    img.onload = () => {
                        this.drawCtx.clearRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
                        this.drawCtx.drawImage(img, 0, 0);
                    };
                    if (this.history[index]) {
                        img.src = this.history[index];
                    }
                }

                undo() {
                    if (this.historyIndex > 0) {
                        this.historyIndex--;
                        this.restoreState(this.historyIndex);
                        this.updateHistoryButtons();
                    }
                }

                redo() {
                    if (this.historyIndex < this.history.length - 1) {
                        this.historyIndex++;
                        this.restoreState(this.historyIndex);
                        this.updateHistoryButtons();
                    }
                }

                updateHistoryButtons() {
                    this.undoBtn.disabled = this.historyIndex < 1;
                    this.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
                }

                resetHistory(initialState = null) {
                    const baseState = initialState || this.drawCanvas.toDataURL("image/png");
                    this.history = [baseState];
                    this.historyIndex = 0;
                    this.updateHistoryButtons();
                }

                handleConnectionChange() {
                    const imageInput = this.node.inputs.find(inp => inp.name === '图像');
                    const isConnected = imageInput && imageInput.link !== null;

                    if (this.loadImageButton) this.loadImageButton.style.display = isConnected ? "inline-block" : "none";

                    if (!isConnected) {
                        this.clearAll();
                    }
                }

                setCanvasSize(width, height) {
                    const w = parseInt(width, 10) || 512;
                    const h = parseInt(height, 10) || 512;

                    [this.bgCanvas, this.drawCanvas, this.previewCanvas].forEach(c => {
                        c.width = w;
                        c.height = h;
                    });

                    if (w > 0 && h > 0) {
                        const aspectRatio = (h / w) * 100;
                        this.canvasContainer.style.height = '0';
                        this.canvasContainer.style.paddingBottom = `${aspectRatio}%`;
                    }
                }

                stopDrawing(e, isMouseLeave = false) {
                    if (!this.drawing) return;
                    const wasDrawing = this.drawing;
                    this.drawing = false;

                    if (!wasDrawing) return;

                    if (this.currentTool === "rect" && !isMouseLeave) {
                        this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                        this.drawSegment(this.drawCtx, this.startPos, this.getMousePos(e));
                    }

                    this.saveState();
                }

                clearDrawingAndHistory() {
                    this.drawCtx.clearRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
                    this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                    this.resetHistory();
                }

                clearAll() {
                    this.bgCtx.clearRect(0, 0, this.bgCanvas.width, this.bgCanvas.height);
                    this.clearDrawingAndHistory();
                }

                finalize() {
                    this.annotationDataWidget.value = this.drawCanvas.toDataURL("image/png");
                    if (this.annotationDataWidget.inputEl) {
                        this.annotationDataWidget.inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }

                startDrawing(e) {
                    this.drawing = true;
                    this.startPos = this.getMousePos(e);
                    this.lastPos = this.startPos;
                }

                draw(e) {
                    if (!this.drawing) return;
                    const pos = this.getMousePos(e);

                    if (this.currentTool === "brush" || this.currentTool === "eraser") {
                        this.drawSegment(this.drawCtx, this.lastPos, pos);
                        this.lastPos = pos;
                    } else if (this.currentTool === "rect") {
                        this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                        this.drawSegment(this.previewCtx, this.startPos, pos);
                    }
                }

                drawSegment(ctx, from, to) {
                    ctx.strokeStyle = this.currentColor;
                    ctx.fillStyle = this.currentColor;
                    ctx.lineWidth = this.brushSize;
                    ctx.globalCompositeOperation = (this.currentTool === 'eraser') ? 'destination-out' : 'source-over';

                    ctx.beginPath();
                    if (this.currentTool === "rect") {
                        ctx.rect(from.x, from.y, to.x - from.x, to.y - from.y);
                        ctx.stroke();
                    } else {
                        ctx.lineCap = "round";
                        ctx.lineJoin = "round";
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.stroke();
                    }
                }

                getMousePos(e) {
                    const rect = this.previewCanvas.getBoundingClientRect();
                    return {
                        x: (e.clientX - rect.left) * (this.drawCanvas.width / rect.width),
                        y: (e.clientY - rect.top) * (this.drawCanvas.height / rect.height)
                    };
                }

                setTool(tool) {
                    this.currentTool = tool;
                    Object.values(this.toolButtons).forEach(btn => btn.classList.remove("active"));
                    if(this.toolButtons[tool]) this.toolButtons[tool].classList.add("active");
                }

                setColor(color) {
                    this.currentColor = color;
                    Object.keys(this.colorBoxes).forEach(k => this.colorBoxes[k].classList.toggle("active", k === color));
                }

                createButton(text, onClick, className = "") {
                    const btn = document.createElement("button");
                    btn.innerText = text;
                    btn.className = `comfy-btn ${className}`;
                    btn.onclick = onClick;
                    return btn;
                }

                createLabel(text) {
                    const lbl = document.createElement("label");
                    lbl.innerText = text;
                    lbl.style.fontSize = "12px";
                    return lbl;
                }

                createSlider(min, max, val, onInput) {
                    const s = document.createElement("input");
                    s.type = "range";
                    s.min = min;
                    s.max = max;
                    s.value = val;
                    s.className = "brush-slider";
                    s.oninput = onInput;
                    return s;
                }

                createColorPalette(colors) {
                    const p = document.createElement("div");
                    p.className = "color-palette";
                    this.colorBoxes = {};

                    colors.forEach(c => {
                        const box = document.createElement("div");
                        box.className = "color-box";
                        box.style.backgroundColor = c;
                        box.onclick = () => this.setColor(c);
                        p.appendChild(box);
                        this.colorBoxes[c] = box;
                    });
                    return p;
                }
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.NAKUAnnotationWidgetV2 = new NAKUAnnotationWidgetV2(this);
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                onConnectionsChange?.apply(this, arguments);
                if (this.inputs?.[index]?.name === '图像') {
                    this.NAKUAnnotationWidgetV2?.handleConnectionChange();
                }
            };
        }
    },
    async setup() {
        if (document.getElementById("naku-annotation-styles-v2")) return;
        const style = document.createElement("style");
        style.id = "naku-annotation-styles-v2";
        style.textContent = `
            .naku-annotation-widget-v2 { display: flex; flex-direction: column; gap: 8px; }
            .naku-annotation-widget-v2 .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
            .naku-annotation-widget-v2 .comfy-btn.active { border-color: #4A90E2; box-shadow: 0 0 3px #4A90E2; }
            .naku-annotation-widget-v2 .comfy-btn.ok-button { background-color: #4A90E2; color: white; }
            .naku-annotation-widget-v2 .color-palette { display: flex; gap: 5px; }
            .naku-annotation-widget-v2 .color-box { width: 24px; height: 24px; border-radius: 4px; cursor: pointer; border: 2px solid transparent; box-sizing: border-box; }
            .naku-annotation-widget-v2 .color-box.active { border-color: #fff; box-shadow: 0 0 5px #000; }
            .naku-annotation-widget-v2 .brush-slider { width: 100px; }
            .naku-annotation-widget-v2 .canvas-container {
                position: relative;
                display: block;
                width: 100%;
                min-height: 100px;
                height: 0;
                padding-bottom: 100%;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                overflow: hidden;
                background-color: #222;
            }
            .naku-annotation-widget-v2 .resizing-canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
        `;
        document.head.appendChild(style);
    }
});


// NAKU Simple Canvas Extension
app.registerExtension({
    name: "Comfy.NAKUSimpleCanvas_NakuNodes",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NAKUSimpleCanvas_NakuNodes") {

            class NAKUSimpleCanvasWidget {
                constructor(node) {
                    this.node = node;
                    this.drawing = false;
                    this.currentTool = "brush";
                    this.currentColor = "#FF0000";  // 默认红色
                    this.brushSize = 15;

                    this.history = [];
                    this.historyIndex = -1;

                    this.element = document.createElement("div");
                    this.element.className = "naku-simple-canvas-widget";
                    this.node.addDOMWidget("simple_canvas_widget", "custom", this.element);

                    // 获取节点上的控件
                    this.canvasDataWidget = node.widgets.find(w => w.name === "canvas_data");
                    this.presetWidget = node.widgets.find(w => w.name === "画板预设");
                    this.orientationWidget = node.widgets.find(w => w.name === "图像模式");
                    this.bgColorWidget = node.widgets.find(w => w.name === "背景颜色");

                    // 隐藏用于数据传输的文本框
                    if (this.canvasDataWidget?.inputEl) {
                        this.canvasDataWidget.inputEl.style.display = "none";
                    }

                    this.setupDOM();
                    this.setupCanvas();
                    this.setupEvents();
                }

                setupDOM() {
                    this.toolbar = document.createElement("div");
                    this.toolbar.className = "toolbar";

                    // 添加提示文字
                    const infoDiv = document.createElement("div");
                    infoDiv.className = "info-text";
                    infoDiv.style.padding = "5px";
                    infoDiv.style.marginBottom = "5px";
                    infoDiv.style.backgroundColor = "#f0f0f0";
                    infoDiv.style.borderRadius = "4px";
                    infoDiv.style.fontSize = "12px";
                    infoDiv.style.color = "#333";
                    infoDiv.innerHTML = "无需连接任何输入接口，请先选择你想要的画板尺寸并点击设置即可在画板上进行使用。";
                    this.element.appendChild(infoDiv);

                    this.toolButtons = {
                        brush: this.createButton("画笔", () => this.setTool("brush")),
                        eraser: this.createButton("橡皮擦", () => this.setTool("eraser")),
                        rect: this.createButton("方框", () => this.setTool("rect")),
                    };

                    this.undoBtn = this.createButton("撤销", () => this.undo());
                    this.redoBtn = this.createButton("重做", () => this.redo());

                    const colorOptions = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF", "#FFFFFF", "#000000"];
                    this.colorPalette = this.createColorPalette(colorOptions);

                    // 设置尺寸按钮 - 使用现有控件的值
                    const setBtn = this.createButton("设置尺寸", () => {
                        const preset = this.presetWidget?.value || "1:1";
                        const orientation = this.orientationWidget?.value || "横屏";
                        this.applyPresetAndOrientation(preset, orientation);
                    });

                    const sliderLabel = this.createLabel("画笔大小:");
                    const brushSlider = this.createSlider(1, 100, this.brushSize, (e) => { this.brushSize = e.target.value; });

                    this.clearBtn = this.createButton("清除", () => this.clearDrawingAndHistory());
                    this.okBtn = this.createButton("确认", () => this.finalize(), "ok-button");

                    this.toolbar.append(
                        this.toolButtons.brush,
                        this.toolButtons.eraser,
                        this.toolButtons.rect,
                        this.undoBtn,
                        this.redoBtn,
                        this.colorPalette,
                        setBtn, // 设置尺寸按钮
                        sliderLabel,
                        brushSlider,
                        this.clearBtn,
                        this.okBtn
                    );

                    this.element.appendChild(this.toolbar);
                    this.setTool("brush");

                    // 监听预设、方向和背景颜色控件的变化
                    if (this.presetWidget && this.presetWidget.inputEl) {
                        this.presetWidget.inputEl.addEventListener('change', () => {
                            const preset = this.presetWidget.value;
                            const orientation = this.orientationWidget?.value || "横屏";
                            this.applyPresetAndOrientation(preset, orientation);
                        });

                        // 隐藏原始控件，因为我们已在自定义工具栏中使用它们
                        if (this.presetWidget.inputEl.parentElement) {
                            this.presetWidget.inputEl.parentElement.style.display = 'none';
                        }
                    }
                    if (this.orientationWidget && this.orientationWidget.inputEl) {
                        this.orientationWidget.inputEl.addEventListener('change', () => {
                            const preset = this.presetWidget?.value || "1:1";
                            const orientation = this.orientationWidget.value;
                            this.applyPresetAndOrientation(preset, orientation);
                        });

                        // 隐藏原始控件，因为我们已在自定义工具栏中使用它们
                        if (this.orientationWidget.inputEl.parentElement) {
                            this.orientationWidget.inputEl.parentElement.style.display = 'none';
                        }
                    }
                    if (this.bgColorWidget && this.bgColorWidget.inputEl) {
                        this.bgColorWidget.inputEl.addEventListener('change', () => {
                            this.updateBackgroundColor(this.bgColorWidget.value);
                        });

                        // 隐藏原始控件，因为我们已在自定义工具栏中使用它们
                        if (this.bgColorWidget.inputEl.parentElement) {
                            this.bgColorWidget.inputEl.parentElement.style.display = 'none';
                        }
                    }
                }

                setupCanvas() {
                    this.canvasContainer = document.createElement("div");
                    this.canvasContainer.className = "canvas-container";

                    this.drawCanvas = document.createElement("canvas");    // 用于显示用户的所有绘制
                    this.previewCanvas = document.createElement("canvas"); // 用于实时预览（如画框）

                    [this.drawCanvas, this.previewCanvas].forEach(c => {
                        c.className = "resizing-canvas";
                        this.canvasContainer.appendChild(c);
                    });

                    this.drawCtx = this.drawCanvas.getContext("2d", { willReadFrequently: true });
                    this.previewCtx = this.previewCanvas.getContext("2d");

                    this.element.appendChild(this.canvasContainer);

                    // 根据预设和方向设置默认尺寸
                    const preset = this.presetWidget?.value || "1:1";
                    const orientation = this.orientationWidget?.value || "横屏";
                    this.applyPresetAndOrientation(preset, orientation);

                    // 设置默认背景颜色
                    const bgColor = this.bgColorWidget?.value || "白色";
                    this.updateBackgroundColor(bgColor);
                }

                applyPresetAndOrientation(preset, orientation) {
                    // 定义预设尺寸
                    const sizeMap = {
                        "1:1": [1328, 1328],
                        "3:2": [1584, 1056],
                        "3:4": [1140, 1472],
                        "16:9": [1664, 928]
                    };

                    let [width, height] = sizeMap[preset] || [512, 512];

                    // 如果是竖屏模式，交换宽高
                    if (orientation === "竖屏") {
                        [width, height] = [height, width];
                    }

                    this.setCanvasSize(width, height);
                }

                updateBackgroundColor(bgColor) {
                    // 定义背景颜色映射
                    const colorMap = {
                        "白色": "#FFFFFF",
                        "黑色": "#000000",
                        "灰色": "#808080"
                    };

                    const backgroundColor = colorMap[bgColor] || "#FFFFFF";
                    this.canvasContainer.style.backgroundColor = backgroundColor;
                }

                setupEvents() {
                    this.previewCanvas.addEventListener("mousedown", (e) => this.startDrawing(e));
                    this.previewCanvas.addEventListener("mousemove", (e) => this.draw(e));
                    this.previewCanvas.addEventListener("mouseup", (e) => this.stopDrawing(e));
                    this.previewCanvas.addEventListener("mouseleave", (e) => this.stopDrawing(e, true));
                }

                saveState() {
                    if (this.historyIndex < this.history.length - 1) {
                        this.history = this.history.slice(0, this.historyIndex + 1);
                    }
                    this.history.push(this.drawCanvas.toDataURL("image/png"));
                    this.historyIndex = this.history.length - 1;
                    this.updateHistoryButtons();
                }

                restoreState(index) {
                    const img = new Image();
                    img.onload = () => {
                        this.drawCtx.clearRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
                        this.drawCtx.drawImage(img, 0, 0);
                    };
                    if (this.history[index]) {
                        img.src = this.history[index];
                    }
                }

                undo() {
                    if (this.historyIndex > 0) {
                        this.historyIndex--;
                        this.restoreState(this.historyIndex);
                        this.updateHistoryButtons();
                    }
                }

                redo() {
                    if (this.historyIndex < this.history.length - 1) {
                        this.historyIndex++;
                        this.restoreState(this.historyIndex);
                        this.updateHistoryButtons();
                    }
                }

                updateHistoryButtons() {
                    this.undoBtn.disabled = this.historyIndex < 1;
                    this.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
                }

                resetHistory(initialState = null) {
                    const baseState = initialState || this.drawCanvas.toDataURL("image/png");
                    this.history = [baseState];
                    this.historyIndex = 0;
                    this.updateHistoryButtons();
                }

                setCanvasSize(width, height) {
                    const w = parseInt(width, 10) || 512;
                    const h = parseInt(height, 10) || 512;

                    [this.drawCanvas, this.previewCanvas].forEach(c => {
                        c.width = w;
                        c.height = h;
                    });

                    if (w > 0 && h > 0) {
                        const aspectRatio = (h / w) * 100;
                        this.canvasContainer.style.height = '0';
                        this.canvasContainer.style.paddingBottom = `${aspectRatio}%`;
                    }
                }

                stopDrawing(e, isMouseLeave = false) {
                    if (!this.drawing) return;
                    const wasDrawing = this.drawing;
                    this.drawing = false;

                    if (!wasDrawing) return;

                    if (this.currentTool === "rect" && !isMouseLeave) {
                        this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                        this.drawSegment(this.drawCtx, this.startPos, this.getMousePos(e));
                    }

                    this.saveState();
                }

                clearDrawingAndHistory() {
                    this.drawCtx.clearRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
                    this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                    this.resetHistory();
                }

                finalize() {
                    this.canvasDataWidget.value = this.drawCanvas.toDataURL("image/png");
                    if (this.canvasDataWidget.inputEl) {
                        this.canvasDataWidget.inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }

                startDrawing(e) {
                    this.drawing = true;
                    this.startPos = this.getMousePos(e);
                    this.lastPos = this.startPos;
                }

                draw(e) {
                    if (!this.drawing) return;
                    const pos = this.getMousePos(e);

                    if (this.currentTool === "brush" || this.currentTool === "eraser") {
                        this.drawSegment(this.drawCtx, this.lastPos, pos);
                        this.lastPos = pos;
                    } else if (this.currentTool === "rect") {
                        this.previewCtx.clearRect(0, 0, this.previewCanvas.width, this.previewCanvas.height);
                        this.drawSegment(this.previewCtx, this.startPos, pos);
                    }
                }

                drawSegment(ctx, from, to) {
                    ctx.strokeStyle = this.currentColor;
                    ctx.fillStyle = this.currentColor;
                    ctx.lineWidth = this.brushSize;
                    ctx.globalCompositeOperation = (this.currentTool === 'eraser') ? 'destination-out' : 'source-over';

                    ctx.beginPath();
                    if (this.currentTool === "rect") {
                        ctx.rect(from.x, from.y, to.x - from.x, to.y - from.y);
                        ctx.stroke();
                    } else {
                        ctx.lineCap = "round";
                        ctx.lineJoin = "round";
                        ctx.moveTo(from.x, from.y);
                        ctx.lineTo(to.x, to.y);
                        ctx.stroke();
                    }
                }

                getMousePos(e) {
                    const rect = this.previewCanvas.getBoundingClientRect();
                    return {
                        x: (e.clientX - rect.left) * (this.drawCanvas.width / rect.width),
                        y: (e.clientY - rect.top) * (this.drawCanvas.height / rect.height)
                    };
                }

                setTool(tool) {
                    this.currentTool = tool;
                    Object.values(this.toolButtons).forEach(btn => btn.classList.remove("active"));
                    if(this.toolButtons[tool]) this.toolButtons[tool].classList.add("active");
                }

                setColor(color) {
                    this.currentColor = color;
                    Object.keys(this.colorBoxes).forEach(k => this.colorBoxes[k].classList.toggle("active", k === color));
                }

                createButton(text, onClick, className = "") {
                    const btn = document.createElement("button");
                    btn.innerText = text;
                    btn.className = `comfy-btn ${className}`;
                    btn.onclick = onClick;
                    return btn;
                }

                createLabel(text) {
                    const lbl = document.createElement("label");
                    lbl.innerText = text;
                    lbl.style.fontSize = "12px";
                    return lbl;
                }

                createSlider(min, max, val, onInput) {
                    const s = document.createElement("input");
                    s.type = "range";
                    s.min = min;
                    s.max = max;
                    s.value = val;
                    s.className = "brush-slider";
                    s.oninput = onInput;
                    return s;
                }

                createColorPalette(colors) {
                    const p = document.createElement("div");
                    p.className = "color-palette";
                    this.colorBoxes = {};

                    colors.forEach(c => {
                        const box = document.createElement("div");
                        box.className = "color-box";
                        box.style.backgroundColor = c;
                        box.onclick = () => this.setColor(c);
                        p.appendChild(box);
                        this.colorBoxes[c] = box;
                    });
                    return p;
                }

                createSelect(options, defaultValue) {
                    const select = document.createElement("select");
                    options.forEach(option => {
                        const optionEl = document.createElement("option");
                        optionEl.value = option;
                        optionEl.text = option;
                        if (option === defaultValue) {
                            optionEl.selected = true;
                        }
                        select.appendChild(optionEl);
                    });
                    return select;
                }
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.NAKUSimpleCanvasWidget = new NAKUSimpleCanvasWidget(this);
            };
        }
    },
    async setup() {
        if (document.getElementById("naku-simple-canvas-styles")) return;
        const style = document.createElement("style");
        style.id = "naku-simple-canvas-styles";
        style.textContent = `
            .naku-simple-canvas-widget { display: flex; flex-direction: column; gap: 8px; }
            .naku-simple-canvas-widget .toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
            .naku-simple-canvas-widget .comfy-btn.active { border-color: #4A90E2; box-shadow: 0 0 3px #4A90E2; }
            .naku-simple-canvas-widget .comfy-btn.ok-button { background-color: #4A90E2; color: white; }
            .naku-simple-canvas-widget .color-palette { display: flex; gap: 5px; }
            .naku-simple-canvas-widget .color-box { width: 24px; height: 24px; border-radius: 4px; cursor: pointer; border: 2px solid transparent; box-sizing: border-box; }
            .naku-simple-canvas-widget .color-box.active { border-color: #fff; box-shadow: 0 0 5px #000; }
            .naku-simple-canvas-widget .brush-slider { width: 100px; }
            .naku-simple-canvas-widget .canvas-container {
                position: relative;
                display: block;
                width: 100%;
                min-height: 100px;
                height: 0;
                padding-bottom: 100%;
                border: 1px solid var(--border-color);
                border-radius: 4px;
                overflow: hidden;
                background-color: #FFFFFF; /* 默认白色背景，将通过JS动态更新 */
            }
            .naku-simple-canvas-widget .resizing-canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            .naku-simple-canvas-widget .info-text {
                background-color: #e6f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 4px;
                padding: 8px;
                margin: 0 0 5px 0;
                font-size: 12px;
                color: #004085;
            }
        `;
        document.head.appendChild(style);
    }
});
