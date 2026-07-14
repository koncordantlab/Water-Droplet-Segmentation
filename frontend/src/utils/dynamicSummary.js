// helper: compute dynamic paragraph similar to backend logic
const computeDynamicSummary = (rowsArr, overlapsObj) => {
if (!rowsArr || rowsArr.length === 0 || !overlapsObj) return null;
const df = rowsArr.slice(); // plain array of objects
// freeze: first frame where ice_cnt >= water_cnt
const freezeRow = df.find((r) => (r.ice_cnt ?? 0) >= (r.water_cnt ?? 0));
const freezeText = freezeRow ? `a majority freeze at approximately ${freezeRow["Frame Number"]} seconds` : "no majority freeze point";
// overlap type
const anyVals = Object.values(overlapsObj).some((v) => v);
let overlapType = "N/A";
if (anyVals) {
    const keys = Object.keys(overlapsObj);
    let maxKey = keys[0];
    for (const k of keys) if ((overlapsObj[k] ?? 0) > (overlapsObj[maxKey] ?? 0)) maxKey = k;
    const map = { ww: "Water-Water", ii: "Ice-Ice", mixed: "Water-Ice" };
    overlapType = `the ${map[maxKey] ?? maxKey} type`;
}
// growth rates from pixel areas
let growthText = "Growth rate could not be calculated.";
if (df.length > 1 && df[0].water_pixel_area !== undefined) {
    let waterDeltas = [];
    let iceDeltas = [];
    for (let i = 1; i < df.length; i++) {
    waterDeltas.push((df[i].water_pixel_area ?? 0) - (df[i - 1].water_pixel_area ?? 0));
    iceDeltas.push((df[i].ice_pixel_area ?? 0) - (df[i - 1].ice_pixel_area ?? 0));
    }
    const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
    const avgWater = avg(waterDeltas);
    const avgIce = avg(iceDeltas);
    growthText = `On average, water area changed by ${avgWater.toFixed(1)} pixels/sec and ice by ${avgIce.toFixed(1)} pixels/sec.`;
}
return `${freezeText}; most interactions were of ${overlapType}. ${growthText}`;
};

export default computeDynamicSummary;