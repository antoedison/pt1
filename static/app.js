// ==============================
// app.js
// Handles both admin.html & chatbot.html
// ==============================

document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------
    // ADMIN PAGE LOGIC
    // -----------------------------
    if (document.body.classList.contains("admin-page")) {
        const uploadForm = document.getElementById("uploadForm");
        const uploadResult = document.getElementById("uploadResult");
        const queryForm = document.getElementById("queryForm");
        const queryResult = document.getElementById("queryResult");

        if (uploadForm) {
            uploadForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                const formData = new FormData(uploadForm);

                try {
                    const res = await fetch("/upload", {
                        method: "POST",
                        body: formData,
                        headers: { Accept: "application/json" },
                    });
                    const data = await res.json();
                    uploadResult.innerHTML = data.message
                        ? `<p class="success">${data.message}</p>`
                        : `<p class="error">${data.error}</p>`;
                } catch (err) {
                    uploadResult.innerHTML = `<p class="error">Upload failed: ${err.message}</p>`;
                }
            });
        }

        if (queryForm) {
            queryForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                const formData = new FormData(queryForm);

                try {
                    const res = await fetch("/query", {
                        method: "POST",
                        body: formData,
                        headers: { Accept: "application/json" },
                    });
                    const data = await res.json();
                    queryResult.innerHTML = data.answer
                        ? `<p><strong>Answer:</strong> ${data.answer}</p>`
                        : `<p class="error">${data.error}</p>`;
                } catch (err) {
                    queryResult.innerHTML = `<p class="error">Query failed: ${err.message}</p>`;
                }
            });
        }
    }

    // -----------------------------
    // CHATBOT PAGE LOGIC
    // -----------------------------
    if (document.body.classList.contains("chat-page")) {
        const chatForm = document.querySelector(".chat-form");
        const chatBox = document.querySelector(".chat-box");
        const inputField = chatForm?.querySelector("input[name='question']");
        const indexSelect = chatForm?.querySelector("select[name='index_name']");

        function addMessage(text, type) {
            const msg = document.createElement("div");
            msg.classList.add("message", type);
            msg.textContent = text;
            chatBox.appendChild(msg);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        if (chatForm) {
            chatForm.addEventListener("submit", async (e) => {
                e.preventDefault();
                const question = inputField.value.trim();
                if (!question) return;

                addMessage(question, "user-msg");
                inputField.value = "";

                try {
                    const formData = new FormData();
                    formData.append("question", question);
                    if (indexSelect) {
                        formData.append("index_name", indexSelect.value);
                    }

                    const res = await fetch("/query", {
                        method: "POST",
                        body: formData,
                        headers: { Accept: "application/json" },
                    });
                    const data = await res.json();

                    if (data.answer) {
                        addMessage(data.answer, "bot-msg");
                    } else {
                        addMessage(data.error || "No response from server", "bot-msg");
                    }
                } catch (err) {
                    addMessage(`Error: ${err.message}`, "bot-msg");
                }
            });
        }
    }
});
