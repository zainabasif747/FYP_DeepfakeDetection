import React, { useState } from "react";
import Header from "./layout/header";
import Footer from "./layout/Footer";
import UploadIcon from "./assests/upload.png";
import axios from "axios";
import { useNavigate } from "react-router-dom"; // Import useNavigate for programmatic navigation

function ImgSec() {
  const [file, setFile] = useState(null); // To store the uploaded file
  const [message, setMessage] = useState(""); // To display backend response
  const [isProcessing, setIsProcessing] = useState(false); // To show processing state
  const [frameCount, setFrameCount] = useState(null); // To store the frame count
  const [frames, setFrames] = useState([]); // To store base64 frames
  const [enhancedImage, setEnhancedImage] = useState(""); // To store enhanced image
  const navigate = useNavigate(); // Use useNavigate for navigation

  // Handle file input change
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setMessage(""); // Clear previous messages
    setFrameCount(null); // Clear previous frame count
    setFrames([]); // Clear previous frames
    setEnhancedImage(""); // Clear previous enhanced image
  };

  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select a file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsProcessing(true); // Set processing state to true

    try {
      // Make API request to backend
      const response = await axios.post("http://127.0.0.1:5000/upload", formData);

      console.log("Backend Response:", response.data);

      // Handle response for videos
      if (response.data.frameCount !== undefined) {
        setFrameCount(response.data.frameCount); // Update frame count for videos
        setFrames(response.data.frames); // Set frames from backend response
        setMessage(`Video processed successfully with ${response.data.frameCount} frames.`);
      } else {
        // Handle enhanced image response
        setEnhancedImage(response.data.enhancedImage); // Set the enhanced image (base64 string)
        setMessage(response.data.message || "File uploaded successfully.");
      }

      // Programmatically navigate to the DisplayResult page with state
      navigate("/DR", {
        state: {
          enhancedImage: response.data.enhancedImage, // Pass the base64 string here
          frameCount: response.data.frameCount,
          frames: response.data.frames,
          message: response.data.message || "File processed successfully.",
        },
      });

    } catch (error) {
      console.error("Error uploading file:", error.response || error);
      setMessage(error.response?.data?.message || "Failed to upload file. Please try again.");
    } finally {
      setIsProcessing(false); // Reset processing state
    }
  };

  return (
    <div className="bg-white min-h-screen flex flex-col overflow-x-hidden">
      <Header />
      <div className="flex-grow flex items-center justify-center">
        <div className="w-full max-w-md text-center">
          <h1 className="text-2xl md:text-3xl font-bold text-blue-600 mb-4">
            Upload and Protect
          </h1>
          <p className="text-gray-600 mb-6">
            Upload and secure your image or video instantly
          </p>
          <div className="relative">
            <input
              type="file"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
              accept="image/*,video/*"
            />
            <div className="absolute inset-y-0 right-4 flex items-center pointer-events-none">
              <img src={UploadIcon} alt="upload" className="h-6" />
            </div>
          </div>
          <button
            onClick={handleUpload}
            className="w-fit mt-4 p-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition duration-300"
            disabled={isProcessing} // Disable button while processing
          >
            {isProcessing ? "Processing..." : "UPLOAD HERE"}
          </button>
          {message && (
            <p className="mt-4 text-sm text-red-600">
              {message}
            </p>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default ImgSec;
