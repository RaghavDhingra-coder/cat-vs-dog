import { useEffect, useRef, useState } from 'react'

// Backend URL comes from frontend/.env (VITE_API_URL), with a sensible default.
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  // Free the preview URL memory when the image changes.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  function selectFile(selected) {
    setResult(null)
    setError('')
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setFile(null)
      setPreviewUrl(null)
      setError('Invalid file. Please choose an image (JPG, PNG, WEBP...).')
      return
    }
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  async function handlePredict() {
    if (!file) {
      setError('No image selected. Please choose an image first.')
      return
    }
    setLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('image', file)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      if (!response.ok) {
        setError(data.error || 'Prediction failed. Please try again.')
      } else {
        setResult(data)
      }
    } catch {
      setError(`Backend unavailable. Make sure the Flask server is running at ${API_URL}.`)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  // Result card style + text for the three possible answers from the backend.
  const resultView = {
    cat: { className: 'cat', title: '🐱 CAT' },
    dog: { className: 'dog', title: '🐶 DOG' },
    not_cat_or_dog: { className: 'other', title: '❓ NOT A CAT OR DOG' },
  }[result?.prediction]

  return (
    <div className="page">
      <header className="header">
        <h1>🐱 Cat vs Dog Classifier 🐶</h1>
        <p>Upload an image and let the CNN identify whether it is a cat or dog.</p>
      </header>

      <main className="card">
        <div
          className={`dropzone ${dragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          {previewUrl ? (
            <img src={previewUrl} alt="Selected preview" className="preview" />
          ) : (
            <div className="dropzone-text">
              <span className="dropzone-icon">📷</span>
              <strong>Click to choose an image</strong>
              <span>or drag and drop it here</span>
            </div>
          )}
        </div>

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => selectFile(e.target.files[0])}
        />

        {file && <p className="file-name">{file.name}</p>}

        <div className="buttons">
          <button className="btn secondary" onClick={() => inputRef.current?.click()} disabled={loading}>
            Choose Image
          </button>
          <button className="btn primary" onClick={handlePredict} disabled={loading}>
            {loading ? 'Analyzing image...' : 'Predict Image'}
          </button>
        </div>

        {error && <div className="error">⚠️ {error}</div>}

        {resultView && (
          <div className={`result ${resultView.className}`}>
            <div className="result-label">Prediction</div>
            <div className="result-value">{resultView.title}</div>
            {result.prediction === 'not_cat_or_dog' && (
              <p className="result-note">No cat or dog was found in this image. Please upload a photo of a cat or a dog.</p>
            )}
            <div className="result-label">Confidence</div>
            <div className="result-confidence">{result.confidence.toFixed(2)}%</div>
            <div className="bar">
              <div className="bar-fill" style={{ width: `${result.confidence}%` }} />
            </div>
          </div>
        )}

        {(result || error || file) && (
          <button className="btn link" onClick={handleReset} disabled={loading}>
            Upload Another Image
          </button>
        )}
      </main>

      <footer className="footer">CNN model built with TensorFlow / Keras · Flask API · React</footer>
    </div>
  )
}

export default App
