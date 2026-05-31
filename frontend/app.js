const predictBtn = document.getElementById('predict');
const resultDiv = document.getElementById('result');
const metricsDiv = document.getElementById('metrics');

predictBtn.addEventListener('click', async () => {
  const payload = {
    data: {
      Age: Number(document.getElementById('age').value || 0),
      MonthlyIncome: Number(document.getElementById('mileage').value || 0),
      TotalWorkingYears: Number(document.getElementById('hp').value || 0),
      YearsAtCompany: Number(document.getElementById('weight').value || 0)
    }
  };

  resultDiv.textContent = 'Predicting...';
  metricsDiv.textContent = '';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    resultDiv.textContent = data.prediction === 1 || data.prediction === 'Yes' || data.prediction === 'yes' ? 'Attrition: YES' : 'Attrition: NO';
    metricsDiv.textContent = `Probability: ${data.probability ?? 'N/A'}`;
  } catch (err) {
    resultDiv.textContent = 'Prediction failed';
    metricsDiv.textContent = err.message;
  }
});
