import { useLocation } from 'react-router-dom';

function DisplayResult() {
  const location = useLocation();
  const { enhancedImage, frameCount, frames, message } = location.state || {};

  // Remove duplicates from the frames array if needed
  const uniqueFrames = Array.from(new Set(frames));

  return (
    <div className="bg-white min-h-screen flex flex-col items-center justify-center">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">Processing Result</h1>
      <p className="text-lg text-gray-600 mb-6">{message}</p>

      {/* Display Enhanced Image if it exists */}
      {enhancedImage && (
        <div className="mb-6">
          <h2 className="text-2xl font-semibold mb-4">Enhanced Image</h2>
          <img
            src={`data:image/png;base64,${enhancedImage}`}
            alt="Enhanced Image"
            className="w-full max-w-md rounded-md shadow-md"
          />
        </div>
      )}

      {/* Display Frames if they exist */}
      {frameCount !== undefined && uniqueFrames.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Frames Processed: {frameCount}</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {uniqueFrames.map((frame, index) => (
              <div
                key={index}
                className="w-full h-64 bg-gray-200 flex justify-center items-center rounded-md shadow-md"
              >
                <img
                  src={`data:image/jpeg;base64,${frame}`}
                  alt={`frame ${index}`}
                  className="object-contain w-full h-full rounded-md"
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default DisplayResult;
