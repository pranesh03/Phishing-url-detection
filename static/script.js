document.getElementById("urlInput").addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevents form submission
        checkURL();
    }
});

async function checkURL() {
    let url = document.getElementById("urlInput").value.trim();
    let resultDiv = document.getElementById("result");

    if (url === "") {
        resultDiv.innerHTML = `<div class='warning'>⚠ Please enter a valid URL!</div>`;
        return;
    }

    try {
        let response = await fetch("/check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url })
        });

        let data = await response.json();

        if (data.status === "fake") {
            resultDiv.innerHTML = `<div class='warning'>⚠ Fake URL detected!</div>`;
        } else {
            resultDiv.innerHTML = `<div class='safe'>✅ URL is safe!</div>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class='error'>❌ Error checking URL. Try again later!</div>`;
    }
}

