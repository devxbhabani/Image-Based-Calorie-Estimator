import React, { useState } from "react";
import "./App.css";

function App() {
	const [selectedFile, setSelectedFile] = useState(null);
	const [previewUrl, setPreviewUrl] = useState(null);
	const [loading, setLoading] = useState(false);
	const [results, setResults] = useState(null);
	const [error, setError] = useState(null);

	const handleFileChange = (e) => {
		const file = e.target.files[0];
		if (file) {
			setSelectedFile(file);
			setPreviewUrl(URL.createObjectURL(file));
			setResults(null);
			setError(null);
		}
	};

	const handleUpload = async (e) => {
		e.preventDefault();
		if (!selectedFile) return;

		setLoading(true);
		setError(null);

		const formData = new FormData();
		formData.append("image", selectedFile);

		try {
			const response = await fetch("/api/estimate", {
				method: "POST",
				body: formData,
			});

			const data = await response.json();
			if (response.ok && data.success) {
				setResults(data);
			} else {
				setError(data.error || "Failed to estimate nutrition.");
			}
			//eslint-disable-next-line
		} catch (err) {
			setError("Connection to backend server failed.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<div className="app-container">
			<header className="navbar">
				<div className="logo-container">
					<span className="logo-emoji">🥗</span>
					<span className="logo-text">AuraCal</span>
				</div>
				<div className="badge">CV & ML Engine</div>
			</header>

			<main className="dashboard-grid">
				{/* Left Control Card */}
				<section className="glass-card upload-section">
					<h2>Analyze Plate</h2>
					<p className="subtitle">
						Upload a top-down food photo to compute structural parameters.
					</p>

					<form onSubmit={handleUpload} className="upload-form">
						<div
							className={`dropzone ${previewUrl ? "has-preview" : ""}`}
						>
							{previewUrl ? (
								<img
									src={previewUrl}
									alt="Meal Preview"
									className="image-preview"
								/>
							) : (
								<div className="dropzone-text">
									<span className="upload-icon">📷</span>
									<p>Drag & drop or click to choose photo</p>
								</div>
							)}
							<input
								type="file"
								accept="image/*"
								onChange={handleFileChange}
								className="file-input"
							/>
						</div>

						<button
							type="submit"
							className="btn-primary"
							disabled={!selectedFile || loading}
						>
							{loading ? (
								<span className="spinner"></span>
							) : (
								"Run Volumetric AI"
							)}
						</button>
					</form>

					{error && <div className="error-alert">{error}</div>}
				</section>

				{/* Right Dashboard Results Card */}
				<section className="glass-card results-section">
					<h2>Estimation Metrics</h2>
					{!results && !loading && (
						<div className="empty-state">
							<p>Please upload a plate photo to load metrics analysis</p>
						</div>
					)}

					{loading && (
						<div className="loading-state">
							<div className="loading-bar"></div>
							<p>Analyzing depth structure...</p>
						</div>
					)}

					{results && (
						<div className="results-content">
							{/* Metrics Grid */}
							<div className="metrics-grid">
								<div className="metric-box">
									<span className="metric-label">
										Estimated Volume
									</span>
									<span className="metric-value">
										{results.volume_cm3}{" "}
										<span className="metric-unit">cm³</span>
									</span>
								</div>
								<div className="metric-box">
									<span className="metric-label">
										Estimated Weight
									</span>
									<span className="metric-value">
										{results.weight_g}{" "}
										<span className="metric-unit">g</span>
									</span>
								</div>
								<div className="metric-box highlighted">
									<span className="metric-label">Energy Load</span>
									<span className="metric-value">
										{results.calories_kcal}{" "}
										<span className="metric-unit">kcal</span>
									</span>
								</div>
							</div>

							{/* Depth Visual Grid */}
							<div className="visuals-grid">
								<div>
									<h4>Input Plate</h4>
									<img
										src={previewUrl}
										alt="Raw Input"
										className="visual-img"
									/>
								</div>
								<div>
									<h4>Depth Estimation Output</h4>
									<img
										src={results.depth_map}
										alt="Depth Render"
										className="visual-img depth-img"
									/>
								</div>
							</div>
						</div>
					)}
				</section>
			</main>
		</div>
	);
}

export default App;
