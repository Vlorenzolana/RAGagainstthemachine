const form = document.getElementById('queryForm');
const gallery = document.getElementById('gallery');
const explanation = document.getElementById('explanation');

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  gallery.innerHTML = '';
  explanation.textContent = 'Consultando…';

  const q = document.getElementById('question').value.trim();
  const count = parseInt(document.getElementById('count').value || '12', 10);

  try {
    const res = await fetch('/api/oracle/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, count })
    });

    if (!res.ok) {
      const txt = await res.text();
      explanation.textContent = `Error: ${res.status} ${txt}`;
      return;
    }

    const data = await res.json();
    explanation.textContent = data.explanation || '';
    renderCards(data.cards || []);
  } catch (err) {
    explanation.textContent = 'Error de red: ' + err.message;
  }
});

function renderCards(cards) {
  gallery.innerHTML = '';
  if (!cards.length) {
    gallery.textContent = 'No hay resultados';
    return;
  }

  for (const c of cards) {
    const el = document.createElement('div');
    el.className = 'card';

    const img = document.createElement('img');
    img.loading = 'lazy';
    img.width = 200;
    img.height = 200;
    img.alt = c.id;
    // prefer thumbnail if present to reduce consumo
    img.src = c.thumbnail || c.url;

    img.addEventListener('click', () => {
      // abrir la imagen original en nueva pestaña (o usar modal sencillo)
      window.open(c.url, '_blank', 'noopener');
    });

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = c.id;

    el.appendChild(img);
    el.appendChild(meta);
    gallery.appendChild(el);
  }
}
