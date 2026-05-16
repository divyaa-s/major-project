import React from "react";
import "../styles/deepfakevideos.css";

const DeepfakeVideos = () => {
  const sampleVideos = [
    { id: 1, title: "Deepfake Example 1", videoId: "WzK1MBEpkJ0" }, // Replace with actual YouTube video ID
    { id: 2, title: "Deepfake Example 2", videoId: "XLWtlUcU5ZI" }, // Replace with actual YouTube video ID
    // Add more YouTube video IDs here
  ];

  return (
    <div className="deepfake-videos-container">
      <h2>Deepfake Video Examples</h2>
      <div className="video-list">
        {sampleVideos.map((video) => (
          <div key={video.id} className="video-item">
            <h3>{video.title}</h3>
            <div className="video-wrapper">
              <iframe
                width="560"
                height="315"
                src={`https://www.youtube.com/embed/${video.videoId}`}
                title={video.title}
                frameBorder="0"
                allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              ></iframe>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DeepfakeVideos;
