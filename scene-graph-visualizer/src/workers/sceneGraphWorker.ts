// src/workers/sceneGraphWorker.ts
// Use Comlink or similar library for easier communication
export {};

const ctx: Worker = self as any;

ctx.onmessage = (e) => {
    const data = e.data;
    // Process data
    const nodes = processSceneGraphData(data);
    ctx.postMessage(nodes);
};

function processSceneGraphData(data: any) {
    // Implement your processing logic here
    return [];
}
