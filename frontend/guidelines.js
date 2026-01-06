function getApiUrl() {
    const el = document.getElementById("apiUrl");
    return el ? el.value.replace(/\/$/, "") : "http://localhost:8000";
}

document.addEventListener("DOMContentLoaded", () => {
    console.log("Guidelines page loaded");
    loadFeatureGuidelines();
});

async function loadFeatureGuidelines() {
    try {
        console.log("Calling API...");
        const res = await fetch(`${getApiUrl()}/features/guidelines`);
        console.log("Response status:", res.status);

        const data = await res.json();
        console.log("Received data:", data);

        const tbody = document.querySelector("#featureTable tbody");
        tbody.innerHTML = "";

        data.rows.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="px-6 py-4 font-mono text-gray-900">
                    ${row.feature || row.feature_name || row["Feature Name"]}
                </td>
                <td class="px-6 py-4 text-gray-700">
                    ${row.description || row["Description"]}
                </td>
            `;
            tbody.appendChild(tr);
        });

        console.log("Table rendered successfully");

    } catch (err) {
        console.error("Feature guideline load failed:", err);
    }
}
