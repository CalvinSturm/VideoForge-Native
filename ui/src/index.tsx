import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

/* --- Global Styles --- */
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";
import "react-mosaic-component/react-mosaic-component.css";
import "./App.css"; // Your custom 3080 Ti UI styles

const rootElement = document.getElementById("root");

if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element. Ensure index.html has <div id='root'></div>");
}