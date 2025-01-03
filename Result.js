import "./App.css";
import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import Header from "../src/components/Header";
import Footer from "../src/components/Footer";

function Result() {
  const location = useLocation();
  const [fileDetails, setFileDetails] = useState(location.state?.fileDetails || {});
  const [realPercentage, setRealPercentage] = useState(65); // Replace with real detection logic

  useEffect(() => {
    // Simulate fetching video duration if it's a video file
    if (fileDetails.fileURL && fileDetails.duration === "Fetching...") {
      const videoElement = document.createElement("video");
      videoElement.src = fileDetails.fileURL;
      videoElement.onloadedmetadata = () => {
        setFileDetails((prev) => ({
          ...prev,
          duration: Math.round(videoElement.duration) + " sec",
        }));
      };
    }
  }, [fileDetails.fileURL, fileDetails.duration]);

  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col items-center">
      {/* Header */}
      <Header />

      <div className="w-full max-w-3xl">
        {/* Heading */}
        <h2 className="text-2xl md:text-3xl font-bold text-blue-600 mb-4 md:mb-6 mt-8">
          Detection Results
        </h2>

        {/* Video/Image Preview */}
        {fileDetails.fileURL && (
          <div className="mb-6">
            {fileDetails.duration !== "N/A" ? (
              <video
                src={fileDetails.fileURL}
                controls
                className="w-full h-48 md:h-64 rounded-md"
              />
            ) : (
              <img
                src={fileDetails.fileURL}
                alt="Uploaded File"
                className="w-full h-48 md:h-64 rounded-md object-cover"
              />
            )}
          </div>
        )}

        {/* Real and Fake Percentages */}
        <div className="flex flex-col md:flex-row items-center justify-around mb-6">
          <div className="flex flex-col items-center mb-6 md:mb-0">
            <div className="w-28 h-28 md:w-32 md:h-32 rounded-full border-[8px] md:border-[10px] border-green-500 flex items-center justify-center">
              <span className="text-2xl md:text-3xl font-bold text-green-600">
                {realPercentage}%
              </span>
            </div>
            <p className="mt-2 text-lg md:text-xl font-semibold">REAL</p>
          </div>
          <div className="flex flex-col items-center">
            <div className="w-28 h-28 md:w-32 md:h-32 rounded-full border-[8px] md:border-[10px] border-red-500 flex items-center justify-center">
              <span className="text-2xl md:text-3xl font-bold text-red-600">
                {100 - realPercentage}%
              </span>
            </div>
            <p className="mt-2 text-lg md:text-xl font-semibold">FAKE</p>
          </div>
        </div>

        {/* File Details */}
        <div className="text-left text-base md:text-lg mb-4">
          <h3 className="text-xl md:text-2xl font-bold mb-3 md:mb-4">Details:</h3>
          <p className="mb-2">
            <span className="font-bold">File name:</span> {fileDetails.name}
          </p>
          <p className="mb-2">
            <span className="font-bold">Duration:</span> {fileDetails.duration}
          </p>
          <p className="mb-4">
            <span className="font-bold">Size:</span> {fileDetails.size}
          </p>
        </div>
      </div>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default Result;