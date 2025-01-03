import "./App.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPaperclip } from "@fortawesome/free-solid-svg-icons";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "../src/components/Header";
import Footer from "../src/components/Footer";

function Detection() {
  const [uploadedFile, setUploadedFile] = useState(null); // Stores the uploaded file details
  const [fileDetails, setFileDetails] = useState(null); // Stores processed file details
  const [error, setError] = useState(""); // Error message
  const navigate = useNavigate();

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const details = {
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + " MB", // File size in MB
        duration: file.type.startsWith("video") ? "Fetching..." : "N/A", // Placeholder for duration
        fileURL: URL.createObjectURL(file), // URL for display
      };

      setUploadedFile(file); // Save file for further processing
      setFileDetails(details); // Save file details
      setError(""); // Clear error message
    }
  };

  // Handle button click to navigate
  const handleUploadClick = () => {
    if (!fileDetails) {
      setError("Please upload a file for detection."); // Show error if no file is selected
      return;
    }
    navigate("/result", { state: { fileDetails } }); // Navigate to result page with file details
  };

  return (
    <div className="flex flex-col justify-between min-h-screen">
      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center flex-grow px-4 md:px-8">
        {/* Heading */}
        <h2 className="text-3xl md:text-4xl font-bold text-blue-600 text-center">
          Scan & Detect
        </h2>
        <p className="text-xl md:text-3xl font-semibold mt-3 text-center">
          Deepfake Videos and Images
        </p>
        <p className="text-gray-600 text-center text-lg md:text-xl mt-4 mb-8">
          Upload videos and images for detection
        </p>

        {/* Upload Field */}
        <div className="relative w-full max-w-md">
          <div className="flex items-center w-full h-16 border border-gray-300 rounded-lg bg-white shadow-sm px-6">
            {/* Vertical Paperclip Icon */}
            <span className="text-black text-3xl mr-4">
              <FontAwesomeIcon icon={faPaperclip} className="transform rotate-[-45deg]" />
            </span>
            <input
              type="file"
              className="absolute inset-0 opacity-0 cursor-pointer"
              accept="video/*,image/*"
              onChange={handleFileUpload}
            />
            <span className="text-gray-500 text-sm md:text-lg flex-grow truncate">
              {fileDetails ? fileDetails.name : "Choose a file"}
            </span>
          </div>
        </div>

        {/* Error Message */}
        {error && <p className="text-red-500 mt-4 text-sm md:text-base">{error}</p>}

        {/* Upload Button */}
        <button
          onClick={handleUploadClick}
          className="mt-8 bg-blue-600 text-white text-lg md:text-xl font-medium py-3 px-10 rounded-lg hover:bg-blue-700 transition"
        >
          DETECT HERE
        </button>
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default Detection;