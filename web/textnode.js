import { app } from "/scripts/app.js";

// NAKU 图片拼接自定义标题编辑界面
app.registerExtension({
    name: "Comfy.NakuNode_图片拼接自定义标题",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "NakuNodeAssetsCombine") {
            
            class NakuCustomTitlesWidget {
                constructor(node) {
                    this.node = node;
                    this.element = document.createElement("div");
                    this.element.className = "naku-custom-titles-widget";
                    this.node.addDOMWidget("custom_titles_editor", "custom", this.element);
                    
                    // 获取相关控件
                    this.useCustomTitlesWidget = node.widgets.find(w => w.name === "use_custom_titles");
                    this.customTitlesJsonWidget = node.widgets.find(w => w.name === "custom_titles_json");
                    
                    // 隐藏 JSON 输入框
                    if (this.customTitlesJsonWidget?.inputEl) {
                        this.customTitlesJsonWidget.inputEl.style.display = "none";
                    }
                    
                    this.setupDOM();
                    this.handleToggleChange();
                }
                
                setupDOM() {
                    // 创建编辑按钮
                    this.editBtn = document.createElement("button");
                    this.editBtn.className = "comfy-btn naku-edit-btn";
                    this.editBtn.textContent = "编辑自定义标题";
                    this.editBtn.onclick = () => this.openEditor();
                    this.element.appendChild(this.editBtn);

                    // 监听开关变化
                    if (this.useCustomTitlesWidget) {
                        const originalCallback = this.useCustomTitlesWidget.callback;
                        this.useCustomTitlesWidget.callback = () => {
                            if (originalCallback) originalCallback.call(this.useCustomTitlesWidget);
                            this.handleToggleChange();
                        };
                    }
                }

                handleToggleChange() {
                    const isEnabled = this.useCustomTitlesWidget?.value;
                    if (this.editBtn) {
                        this.editBtn.disabled = !isEnabled;
                        this.editBtn.style.opacity = isEnabled ? "1" : "0.5";
                    }
                }
                
                openEditor() {
                    // 解析当前 JSON
                    let currentTitles = {};
                    try {
                        if (this.customTitlesJsonWidget?.value) {
                            currentTitles = JSON.parse(this.customTitlesJsonWidget.value);
                        }
                    } catch (e) {
                        console.error("解析标题 JSON 失败:", e);
                    }
                    
                    // 创建模态框
                    const modal = document.createElement("div");
                    modal.className = "naku-titles-modal-overlay";
                    
                    const modalContent = document.createElement("div");
                    modalContent.className = "naku-titles-modal";
                    
                    // 标题栏
                    const headerEl = document.createElement("div");
                    headerEl.className = "naku-titles-modal-header";
                    headerEl.innerHTML = `
                        <h3 style="margin:0;">自定义图片标题</h3>
                        <button class="naku-close-btn" onclick="this.closest('.naku-titles-modal-overlay').remove()">×</button>
                    `;
                    modalContent.appendChild(headerEl);
                    
                    // 输入框区域
                    const bodyEl = document.createElement("div");
                    bodyEl.className = "naku-titles-modal-body";
                    
                    // 创建 9 个输入框
                    const defaultLabels = [
                        "图片 1 (Front)",
                        "图片 2 (Left)", 
                        "图片 3 (Right)",
                        "图片 4 (High Angle)",
                        "图片 5 (Low Angle)",
                        "图片 6 (Back)",
                        "图片 7 (Back Side)",
                        "图片 8 (Detail 1)",
                        "图片 9 (Detail 2)"
                    ];
                    
                    const inputs = [];
                    for (let i = 0; i < 9; i++) {
                        const rowEl = document.createElement("div");
                        rowEl.className = "naku-title-row";
                        
                        const labelEl = document.createElement("label");
                        labelEl.textContent = `图片${i + 1}:`;
                        labelEl.className = "naku-title-label";
                        
                        const inputEl = document.createElement("input");
                        inputEl.type = "text";
                        inputEl.className = "comfy-input naku-title-input";
                        inputEl.placeholder = `图片${i + 1} 的标题`;
                        inputEl.value = currentTitles[String(i)] || defaultLabels[i];
                        
                        inputs.push(inputEl);
                        
                        rowEl.appendChild(labelEl);
                        rowEl.appendChild(inputEl);
                        bodyEl.appendChild(rowEl);
                    }
                    
                    modalContent.appendChild(bodyEl);
                    
                    // 按钮区域
                    const footerEl = document.createElement("div");
                    footerEl.className = "naku-titles-modal-footer";
                    
                    const cancelBtn = document.createElement("button");
                    cancelBtn.className = "comfy-btn";
                    cancelBtn.textContent = "取消";
                    cancelBtn.onclick = () => modal.remove();
                    
                    const confirmBtn = document.createElement("button");
                    confirmBtn.className = "comfy-btn naku-confirm-btn";
                    confirmBtn.textContent = "确认";
                    confirmBtn.onclick = () => {
                        // 构建 JSON 对象
                        const titles = {};
                        inputs.forEach((input, index) => {
                            if (input.value.trim()) {
                                titles[String(index)] = input.value.trim();
                            }
                        });
                        
                        // 更新 JSON 控件的值
                        if (this.customTitlesJsonWidget) {
                            this.customTitlesJsonWidget.value = JSON.stringify(titles, null, 2);
                            // 触发 change 事件
                            if (this.customTitlesJsonWidget.inputEl) {
                                this.customTitlesJsonWidget.inputEl.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        
                        // 更新预览
                        this.updatePreview();
                        
                        modal.remove();
                    };
                    
                    footerEl.appendChild(cancelBtn);
                    footerEl.appendChild(confirmBtn);
                    modalContent.appendChild(footerEl);
                    
                    modal.appendChild(modalContent);
                    document.body.appendChild(modal);
                }
            }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.NakuCustomTitlesWidget = new NakuCustomTitlesWidget(this);
            };
        }
    },
    async setup() {
        if (document.getElementById("naku-custom-titles-styles")) return;
        const style = document.createElement("style");
        style.id = "naku-custom-titles-styles";
        style.textContent = `
            .naku-custom-titles-widget { 
                display: block;
                padding: 0;
                margin: 0;
                width: 100%;
                box-sizing: border-box;
            }
            .naku-custom-titles-widget * {
                box-sizing: border-box;
            }
            .naku-edit-btn {
                background-color: #4A90E2;
                color: white;
                font-weight: bold;
                padding: 5px 8px;
                border-radius: 3px;
                border: none;
                cursor: pointer;
                font-size: 11px;
                width: 100%;
                text-align: center;
                margin: 0;
            }
            .naku-edit-btn:disabled {
                background-color: #666;
                cursor: not-allowed;
                opacity: 0.6;
            }
            .naku-edit-btn:hover:not(:disabled) {
                background-color: #3a7bc8;
            }
            .naku-titles-modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10000;
            }
            .naku-titles-modal {
                background-color: var(--comfy-menu-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 16px;
                min-width: 400px;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            }
            .naku-titles-modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 1px solid var(--border-color);
            }
            .naku-titles-modal-header h3 {
                color: var(--fg-color);
                font-size: 16px;
            }
            .naku-close-btn {
                background: none;
                border: none;
                color: var(--fg-color);
                font-size: 24px;
                cursor: pointer;
                padding: 0 4px;
                line-height: 1;
            }
            .naku-close-btn:hover {
                color: #ff4444;
            }
            .naku-titles-modal-body {
                display: flex;
                flex-direction: column;
                gap: 12px;
                margin-bottom: 16px;
            }
            .naku-title-row {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            .naku-title-label {
                min-width: 80px;
                color: var(--fg-color);
                font-size: 13px;
            }
            .naku-title-input {
                flex: 1;
                padding: 6px 8px;
                background-color: var(--input-bg);
                border: 1px solid var(--border-color);
                border-radius: 4px;
                color: var(--fg-color);
                font-size: 13px;
            }
            .naku-title-input:focus {
                outline: none;
                border-color: #4A90E2;
            }
            .naku-titles-modal-footer {
                display: flex;
                justify-content: flex-end;
                gap: 8px;
                padding-top: 8px;
                border-top: 1px solid var(--border-color);
            }
            .naku-confirm-btn {
                background-color: #4A90E2;
                color: white;
            }
        `;
        document.head.appendChild(style);
    }
});
