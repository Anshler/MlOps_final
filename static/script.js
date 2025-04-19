const form = document.querySelector('form');
const tableBody = document.querySelector('#prediction-table tbody');
const accuracyDiv = document.getElementById('accuracy');
const expandBtn = document.getElementById('expand-btn');

let allLabels = [];
let allPredictions = [];
let currentIndex = 0;
const pageSize = 100;

function renderTableChunk() {
    const end = Math.min(currentIndex + pageSize, allLabels.length);
    for (let i = currentIndex; i < end; i++) {
        const tr = document.createElement('tr');
        const isCorrect = allLabels[i] === allPredictions[i];
        tr.className = isCorrect ? 'correct' : 'incorrect';
        tr.innerHTML = `
            <td>${i}</td>
            <td>${allLabels[i]}</td>
            <td>${allPredictions[i]}</td>
        `;
        tableBody.appendChild(tr);
    }
    currentIndex = end;

    expandBtn.style.display = currentIndex < allLabels.length ? 'inline-block' : 'none';
}

form.onsubmit = async function(event) {
    event.preventDefault();
    const formData = new FormData(form);
    const response = await fetch('/generate', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();

    allLabels = result.labels;
    allPredictions = result.predictions;
    currentIndex = 0;
    tableBody.innerHTML = '';
    accuracyDiv.innerHTML = `<strong>Accuracy:</strong> ${result.accuracy}`;

    renderTableChunk();
};

expandBtn.onclick = renderTableChunk;
