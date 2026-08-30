import { app } from "../../scripts/app.js";

const LOADER_NODE_NAME = "MiniMaxH3EnhancedLoaderStar7";
const VSA_NODE_NAME = "MiniMaxH3VSASwitchStar7";

function isChinese() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language
        ?? "en";
    return String(locale).toLowerCase().startsWith("zh");
}

function localizeLoader(node) {
    const chinese = isChinese();
    node.title = chinese
        ? "MiniMax H3 自适应优化载入 - Star7"
        : "MiniMax H3 Enhanced Loader - Star7";

    const modelWidget = node.widgets?.find((widget) => widget.name === "unet_name");
    if (modelWidget) {
        modelWidget.label = chinese ? "H3 模型" : "H3 model";
    }
    const output = node.outputs?.find((item) => item.name === "model");
    if (output) {
        output.localized_name = chinese ? "模型" : "model";
    }
}

function localizeVsaSwitch(node) {
    const chinese = isChinese();
    node.title = chinese
        ? "MiniMax H3 FastH3 VSA 加速 - Star7"
        : "MiniMax H3 FastH3 VSA Switch - Star7";
    const switchWidget = node.widgets?.find(
        (widget) => widget.name === "vsa_acceleration"
    );
    if (switchWidget) {
        switchWidget.label = chinese ? "VSA 加速" : "VSA acceleration";
    }
    const input = node.inputs?.find((item) => item.name === "model");
    if (input) {
        input.localized_name = chinese ? "模型" : "model";
    }
    const output = node.outputs?.find((item) => item.name === "model");
    if (output) {
        output.localized_name = chinese ? "模型" : "model";
    }
}

app.registerExtension({
    name: "Star7.MiniMaxH3EnhancedLoader.Localization",
    nodeCreated(node) {
        if (node.comfyClass === LOADER_NODE_NAME) {
            localizeLoader(node);
        } else if (node.comfyClass === VSA_NODE_NAME) {
            localizeVsaSwitch(node);
        }
    },
});
