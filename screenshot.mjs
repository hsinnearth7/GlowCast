import { chromium } from 'playwright';
import path from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync, statSync } from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const demoDir = path.join(__dirname, 'demo', 'screenshots');
mkdirSync(demoDir, { recursive: true });
const BASE = 'http://localhost:8501';

const pages = [
  { name: '01_overview', label: 'Executive Overview' },
  { name: '02_forecasting', label: 'Demand & Forecasting' },
  { name: '03_supply_chain', label: 'Inventory & Supply Chain' },
  { name: '04_causal', label: 'Causal & Experimentation' },
  { name: '05_mlops', label: 'MLOps & Quality' },
];

async function waitForStreamlit(page) {
  await page.waitForSelector('[data-testid="stSidebar"]', { timeout: 30000 });
  await page.waitForTimeout(3000);
}

(async () => {
  const browser = await chromium.launch();

  for (const pg of pages) {
    console.log(`Capturing: ${pg.name}...`);
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });

    await page.goto(BASE, { waitUntil: 'networkidle', timeout: 30000 });
    await waitForStreamlit(page);

    // Click sidebar button to navigate (buttons contain the page label text)
    const navButton = page.locator(`[data-testid="stSidebar"]`)
      .locator('button', { hasText: pg.label });

    if (await navButton.count() > 0) {
      await navButton.first().click();
      await page.waitForTimeout(5000);
    }

    // Wait for plotly charts to render
    await page.waitForTimeout(3000);

    const filePath = path.join(demoDir, `${pg.name}.png`);
    await page.screenshot({ path: filePath, fullPage: true });
    const size = statSync(filePath).size;
    console.log(`  OK: ${(size / 1024).toFixed(1)} KB`);
    await page.close();
  }

  await browser.close();
  console.log('\nAll screenshots saved to demo/screenshots/');
})();
