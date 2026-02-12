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
            <div className="submit-section">
                <button
                    id="run-btn"
                    onClick={(ev) => handleSubmit(ev, videoPath, isSegmentationOn)}
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
