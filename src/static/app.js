const body = document.querySelector("#ordersBody");
const simulateBtn = document.querySelector("#simulateBtn");
const modelStatus = document.querySelector("#modelStatus");
const ordersCount = document.querySelector("#ordersCount");
const expectedRevenue = document.querySelector("#expectedRevenue");
const highPriority = document.querySelector("#highPriority");
const highRisk = document.querySelector("#highRisk");

const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const percent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`Error ${response.status}`);
  return response.json();
}

function renderRows(orders) {
  if (!orders.length) {
    body.innerHTML = '<tr><td colspan="6" class="empty">Sin pedidos todavia</td></tr>';
    return;
  }

  body.innerHTML = orders
    .map(
      (order) => `
      <tr>
        <td><strong>${order.order_id}</strong><br><span>${order.category}</span></td>
        <td>${order.product_name}</td>
        <td>${money.format(order.predicted_order_value_usd)}</td>
        <td>${percent.format(order.return_probability)}</td>
        <td>${money.format(order.expected_revenue_usd)}</td>
        <td><span class="pill ${order.shipping_priority}">${order.shipping_priority}</span></td>
      </tr>
    `,
    )
    .join("");
}

function renderMetrics(orders) {
  const totalRevenue = orders.reduce((sum, order) => sum + Number(order.expected_revenue_usd || 0), 0);
  ordersCount.textContent = orders.length;
  expectedRevenue.textContent = money.format(totalRevenue);
  highPriority.textContent = orders.filter((order) => order.shipping_priority === "High").length;
  highRisk.textContent = orders.filter((order) => order.return_risk === "high").length;
}

async function loadOrders() {
  const orders = await fetchJson("/api/orders?limit=25");
  renderRows(orders);
  renderMetrics(orders);
}

async function loadModelInfo() {
  const info = await fetchJson("/api/model-info");
  modelStatus.textContent = `${info.model_type} | ${info.train_rows} filas de entrenamiento`;
}

simulateBtn.addEventListener("click", async () => {
  simulateBtn.disabled = true;
  simulateBtn.textContent = "Procesando...";
  try {
    await fetchJson("/api/orders/simulate", { method: "POST" });
    await loadOrders();
  } finally {
    simulateBtn.disabled = false;
    simulateBtn.textContent = "Simular pedido";
  }
});

loadModelInfo().catch(() => {
  modelStatus.textContent = "No se pudo cargar informacion del modelo";
});
loadOrders();
