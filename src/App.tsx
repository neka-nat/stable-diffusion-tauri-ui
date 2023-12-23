import { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";

function App() {
  const [imageData, setImageData] = useState("");
  const [prompt, setPrompt] = useState("");

  const handleGenerateImage = async () => {
    try {
      const data = await invoke('generate', { prompt });
      setImageData(`data:image/png;base64,${data}`);
    } catch (error) {
      console.error('Error generating image:', error);
    }
  };

  return (
    <div className="container">
      <h1>Stable diffusion WebGPU</h1>

      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          handleGenerateImage();
        }}
      >
        <input
          id="generate-input"
          onChange={(e) => setPrompt(e.currentTarget.value)}
          placeholder="Enter a prompt..."
        />
        <button type="submit">Generate</button>
      </form>

      {imageData && <img src={imageData} alt="Generated" />}
    </div>
  );
}

export default App;
