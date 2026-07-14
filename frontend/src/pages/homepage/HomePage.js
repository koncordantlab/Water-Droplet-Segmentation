import { useState } from "react"

import "./homePage.css"

import cloudImage from '../../assets/cloud_upload1.png'
import iceImage from '../../assets/Ice_image.jpg'

const HomePage = ({
    loading,
    handleSubmit,
    downloadUrl,
    error
}) => {

    const [videoPath, setVideoPath] = useState("")
    const [isSegmentationOn, setIsSegmentationOn] = useState(true)
    const [distInterval, setDistInterval] = useState(0)
    const [outputMode, setOutputMode] = useState("full")
    const [umPerPx, setUmPerPx] = useState("")

    const handleTextInputChange = (e) => {
        setVideoPath(e.target.value)
    }

    const handlebrowserPrompt = () => {
        const p = window.prompt("Enter video file path:");
        if (p !== null && typeof p === 'string') {
            setVideoPath(p);
        }
    }
    
  return (
    <div className='home-page-container'>
      <div className='left-section-input'>
        <div className="headers">
            <h1 id="project-header">NASA</h1>
            <h2 className="project-descriptor">Video Processing Unit</h2>
        </div>
        <div className="input-section">
            <button 
                className="browser-input-field"
                onClick={handlebrowserPrompt}
            >
                <img 
                    src={cloudImage}
                    alt="browse"
                    style={{ width: 150 }}
                />
            </button>

            <div className="text-input-container">
                <input 
                    id="text-input-field"
                    type="text"
                    placeholder="Paste video file path here"
                    value={videoPath}
                    onChange={handleTextInputChange}
                />
            </div>
        </div>

        <div className="bottom-section">
            <div className="segmentation-switch">
                <label className="switch">
                    <input 
                        id="overlay-toggle"
                        type="checkbox"
                        checked={isSegmentationOn}
                        onChange={() => setIsSegmentationOn(!isSegmentationOn)}
                    />
                    <span className="slider-switch"></span>
                </label>
                <span className="switch-text">Save Segmentation overlay video</span>
            </div>
            <div className="dist-interval-input" style={{ marginTop: 12 }}>
                <label htmlFor="dist-interval-field" className="switch-text" style={{ marginRight: 8 }}>
                    Droplet size distribution every N frames (0 = off):
                </label>
                <input
                    id="dist-interval-field"
                    type="number"
                    min={0}
                    step={1}
                    value={distInterval}
                    onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        setDistInterval(Number.isNaN(v) || v < 0 ? 0 : v);
                    }}
                    style={{ width: 80 }}
                />
            </div>
            <div className="output-mode-input" style={{ marginTop: 12 }}>
                <label htmlFor="output-mode-field" className="switch-text" style={{ marginRight: 8 }}>
                    Output detail:
                </label>
                <select
                    id="output-mode-field"
                    value={outputMode}
                    onChange={(e) => setOutputMode(e.target.value)}
                >
                    <option value="full">Full (all metrics)</option>
                    <option value="basic">Basic (sizes only)</option>
                </select>
            </div>
            <div className="um-per-px-input" style={{ marginTop: 12 }}>
                <label htmlFor="um-per-px-field" className="switch-text" style={{ marginRight: 8 }}>
                    Scale (µm per pixel, optional):
                </label>
                <input
                    id="um-per-px-field"
                    type="number"
                    min={0}
                    step="any"
                    value={umPerPx}
                    onChange={(e) => setUmPerPx(e.target.value)}
                    style={{ width: 80 }}
                />
            </div>
            <div className="submit-section">
                <button
                    id="run-btn"
                    onClick={(ev) => handleSubmit(ev, videoPath, isSegmentationOn, distInterval, outputMode, umPerPx)}
                    disabled={loading || videoPath === ""}
                >
                    {loading ? "Processing..." : "Run Detection"}
                </button>
                {downloadUrl && (
                    <div style={{ marginTop: 8 }}>
                        <a href={downloadUrl} target="_blank" rel="noreferrer" style={{ color: "#e6c645" }}>Download Latest Excel</a>
                    </div>
                )}
            </div>
        </div>
      </div>
      <div className="right-section-image">
        <img 
            src={iceImage}
            alt="ice"
            className="ice-image"
        />
      </div>
    </div>
  )
}

export default HomePage
