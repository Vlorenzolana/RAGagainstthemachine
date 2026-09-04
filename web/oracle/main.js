const form = document.getElementById("oracle-form");
const result = document.getElementById("result");
const explanation = document.getElementById("explanation");
const cardsContainer = document.getElementById("cards");
const errorEl = document.getElementById("error");

function renderCards(cards) {
  cardsContainer.innerHTML = "";
  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "card";

    const img = document.createElement("img");
    img.src = card.thumbnail;
    img.alt = card.id;
    img.loading = "lazy";

    const caption = document.createElement("p");
    caption.textContent = card.id;

    article.appendChild(img);
    article.appendChild(caption);
    cardsContainer.appendChild(article);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorEl.textContent = "";

  const payload = {
    question: document.getElementById("question").value.trim(),
    count: Number(document.getElementById("count").value || 12),
  };

  try {
    const response = await fetch("/api/oracle/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }

    const data = await response.json();
    explanation.textContent = data.explanation;
    renderCards(data.cards || []);
    result.classList.remove("hidden");
  } catch (error) {
    result.classList.add("hidden");
    errorEl.textContent = error.message || "Unknown error";
  }
});
