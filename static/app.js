const form = document.getElementById("form");
const result = document.getElementById("result");
const verdict = document.getElementById("verdict");
const fill = document.getElementById("fill");
const prob = document.getElementById("prob");
const NUMBERS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"];

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form));
  NUMBERS.forEach((k) => (data[k] = Number(data[k])));

  const btn = form.querySelector("button");
  btn.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Geçersiz veri veya sunucu hatası");
    const { prediction, probability } = await res.json();

    const high = prediction === 1;
    result.className = high ? "high" : "low";
    verdict.textContent = high ? "⚠️ Yüksek kalp hastalığı riski" : "✅ Düşük kalp hastalığı riski";
    prob.textContent = `Hastalık olasılığı: %${(probability * 100).toFixed(1)}`;
    fill.style.width = "0";
    requestAnimationFrame(() => (fill.style.width = `${probability * 100}%`));
  } catch (err) {
    result.className = "high";
    verdict.textContent = err.message;
    prob.textContent = "";
    fill.style.width = "0";
  } finally {
    btn.disabled = false;
  }
});
