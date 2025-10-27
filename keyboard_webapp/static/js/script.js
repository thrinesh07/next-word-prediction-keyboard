document.addEventListener("DOMContentLoaded", () => {
  const inputText = document.getElementById("inputText");
  const suggestionButtons = document.querySelectorAll(".sugg-btn");
  const tempRange = document.getElementById("temperature");
  const tempValue = document.getElementById("tempValue");

  let debounceTimer = null;

  async function fetchSuggestions(text) {
    const resp = await fetch("/api/suggest", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        seed_text: text,
        temperature: parseFloat(tempRange.value)
      })
    });
    return resp.json();
  }

  async function updateSuggestions() {
    const text = inputText.value.trim();
    if (!text) {
      suggestionButtons.forEach(b => {
        b.textContent = "...";
        b.disabled = true;
      });
      return;
    }

    const { suggestions } = await fetchSuggestions(text);
    suggestions.forEach((s, i) => {
      if (suggestionButtons[i]) {
        suggestionButtons[i].textContent = s[0] || "...";
        suggestionButtons[i].disabled = false;
      }
    });
  }

  inputText.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(updateSuggestions, 400);
  });

  suggestionButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      if (btn.disabled || btn.textContent === "...") return;
      inputText.value = inputText.value.trim() + " " + btn.textContent + " ";
      updateSuggestions();
    });
  });

  tempRange.addEventListener("input", () => {
    tempValue.textContent = tempRange.value;
    updateSuggestions();
  });
});
