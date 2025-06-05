/**
 * Vanilla Least Squares Linear Regression for Concrete Strength Prediction
 * Loads CSV, splits data, fits model, evaluates accuracy.
 * No dependencies, for educational reference.
 * qodo GPT-4.1
 */

document.getElementById('csvFile').addEventListener('change', handleFile, false);

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const csv = e.target.result;
    const data = parseCSV(csv);
    runLeastSquaresDemo(data);
  };
  reader.readAsText(file);
}

// Simple CSV parser (assumes no quoted fields, no commas in data)
function parseCSV(csv) {
  const lines = csv.trim().split('\n');
  const header = lines[0].split(',');
  const rows = lines.slice(1).map(line => line.split(',').map(Number));
  return { header, rows };
}

// Split data into train/test (80/20 split)
function trainTestSplit(rows, testRatio = 0.2) {
  const shuffled = rows.slice().sort(() => Math.random() - 0.5);
  const testCount = Math.floor(rows.length * testRatio);
  const test = shuffled.slice(0, testCount);
  const train = shuffled.slice(testCount);
  return { train, test };
}

// Add a column of 1s for bias term
function addBias(X) {
  return X.map(row => [1, ...row]);
}

// Matrix transpose
function transpose(A) {
  return A[0].map((_, i) => A.map(row => row[i]));
}

// Matrix multiplication
function matMul(A, B) {
  const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
  for (let i = 0; i < A.length; i++) {
    for (let j = 0; j < B[0].length; j++) {
      for (let k = 0; k < B.length; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

// Matrix inverse (only for small square matrices, using Gauss-Jordan)
function matInv(A) {
  const n = A.length;
  const I = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));
  const M = A.map(row => row.slice());
  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    // Swap rows
    [M[i], M[maxRow]] = [M[maxRow], M[i]];
    [I[i], I[maxRow]] = [I[maxRow], I[i]];
    // Normalize row
    const pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) throw new Error("Matrix not invertible");
    for (let j = 0; j < n; j++) {
      M[i][j] /= pivot;
      I[i][j] /= pivot;
    }
    // Eliminate other rows
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = 0; j < n; j++) {
        M[k][j] -= factor * M[i][j];
        I[k][j] -= factor * I[i][j];
      }
    }
  }
  return I;
}

// Least squares fit: beta = (X^T X)^-1 X^T y
function leastSquaresFit(X, y) {
  const XT = transpose(X);
  const XTX = matMul(XT, X);
  const XTy = matMul(XT, y.map(v => [v]));
  const XTXinv = matInv(XTX);
  const beta = matMul(XTXinv, XTy).map(row => row[0]);
  return beta;
}

// Predict y given X and beta
function predict(X, beta) {
  var predictions = [];
  for (var i = 0; i < X.length; i++) {
    var row = X[i];
    var sum = 0;
    for (var j = 0; j < row.length; j++) {
      sum += row[j] * beta[j];
    }
    predictions.push(sum);
  }
  //return X.map(row => row.reduce((sum, xj, j) => sum + xj * beta[j], 0)); // A more concise way to do this in JS
  return predictions;
}

// Compute RMSE
function rmse(yTrue, yPred) {
  const n = yTrue.length;
  const mse = yTrue.reduce((sum, y, i) => sum + (y - yPred[i]) ** 2, 0) / n;
  return Math.sqrt(mse);
}

// Main demo logic
function runLeastSquaresDemo(data) {
  // Features: all columns except last (strength)
  // Target: last column
  const Xall = data.rows.map(row => row.slice(0, -1));
  const yall = data.rows.map(row => row[row.length - 1]);
  // Split
  const { train, test } = trainTestSplit(data.rows, 0.2);
  const Xtrain = train.map(row => row.slice(0, -1));
  const ytrain = train.map(row => row[row.length - 1]);
  const Xtest = test.map(row => row.slice(0, -1));
  const ytest = test.map(row => row[row.length - 1]);
  // Add bias
  const XtrainBias = addBias(Xtrain);
  const XtestBias = addBias(Xtest);
  // Fit
  let beta;
  try {
    beta = leastSquaresFit(XtrainBias, ytrain);
  } catch (e) {
    document.getElementById('output').textContent = "Error: " + e.message;
    return;
  }
  // Predict
  const ytrainPred = predict(XtrainBias, beta);
  const ytestPred = predict(XtestBias, beta);
  // Evaluate
  const trainRMSE = rmse(ytrain, ytrainPred);
  const testRMSE = rmse(ytest, ytestPred);

  // Output
  const output = [
    "Least Squares Linear Regression (Vanilla JS)",
    "Features: " + data.header.slice(0, -1).join(', '),
    "Target: " + data.header[data.header.length - 1],
    "",
    `Train size: ${train.length}, Test size: ${test.length}`,
    "",
    "Coefficients (bias first):",
    beta.map((b, i) => `  ${i === 0 ? 'bias' : data.header[i-1]}: ${b.toFixed(4)}`).join('\n'),
    "",
    `Train RMSE: ${trainRMSE.toFixed(3)}`,
    `Test RMSE:  ${testRMSE.toFixed(3)}`,
    "",
    "Sample predictions (test set):",
    ...ytest.slice(0, 10).map((y, i) =>
      `  True: ${y.toFixed(2)}, Pred: ${ytestPred[i].toFixed(2)}`
    )
  ].join('\n');
  document.getElementById('output').textContent = output;
}
