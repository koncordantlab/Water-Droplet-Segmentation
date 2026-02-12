const buildPercentageFigure = (c, sliderValue) => {
    if (!c?.pct) return null;
    // slice data to sliderValue (assumes x is an array of frame numbers / indices)
    const limit = Math.max(1, Math.min(Number(sliderValue || 1), (c.pct.x || []).length));
    const sliceIdx = (arr) => (Array.isArray(arr) ? arr.slice(0, limit) : arr);
    const x = sliceIdx(c.pct.x);
    const water = sliceIdx(c.pct.water);
    const ice = sliceIdx(c.pct.ice);
    const layout = { title: `Water & Ice (%) (first ${sliderValue} frames)`, margin: { t: 40 } };
    if (Array.isArray(x) && x.length > 0) layout.xaxis = { range: [x[0], x[x.length - 1]] };
    return {
      data: [
        { x, y: water, type: "scatter", mode: "lines+markers", name: "Water (%)" },
        { x, y: ice, type: "scatter", mode: "lines+markers", name: "Ice (%)" },
      ],
      layout,
      config: { responsive: true },
    };
}

const buildOverlappingFigure = (c, sliderValue) => {
    if (!c?.ov) return null;
    const limit = Math.max(1, Math.min(Number(sliderValue || 1), (c.ov.x || []).length));
    const sliceIdx = (arr) => (Array.isArray(arr) ? arr.slice(0, limit) : arr);
    const x = sliceIdx(c.ov.x);
    const ww = sliceIdx(c.ov.ww);
    const ii = sliceIdx(c.ov.ii);
    const wi = sliceIdx(c.ov.wi);
    const layout = { title: `Overlaps (first ${sliderValue} frames)`, margin: { t: 40 } };
    if (Array.isArray(x) && x.length > 0) layout.xaxis = { range: [x[0], x[x.length - 1]] };
    return {
      data: [
        { x, y: ww, type: "scatter", mode: "lines+markers", name: "Water–Water" },
        { x, y: ii, type: "scatter", mode: "lines+markers", name: "Ice–Ice" },
        { x, y: wi, type: "scatter", mode: "lines+markers", name: "Water–Ice" },
      ],
      layout,
      config: { responsive: true },
    };
}

const donutFigure = (labels, values, title) => ({
    data: [{ labels, values, type: "pie", hole: 0.0 }],
    layout: { 
      title: {text: title, font: {size:16}}, 
      showlegend: true, 
      margin: { t: 30 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {color: "#fff"},
    },
    config: { 
      responsive: true,
      displayModeBar: false
    },
  });

export { buildOverlappingFigure, buildPercentageFigure, donutFigure };

