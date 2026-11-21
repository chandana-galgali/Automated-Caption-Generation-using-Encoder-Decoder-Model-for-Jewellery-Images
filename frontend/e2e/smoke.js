// End-to-end test of the built website in a real (headless) Chromium.
// Usage: node smoke.js <site-url> <path-to-a-test-image>
const { chromium } = require('playwright');

(async () => {
  const [url, imagePath] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

  const t0 = Date.now();
  await page.goto(url);
  await page.getByText('Model ready', { exact: false }).waitFor({ timeout: 240000 });
  console.log(`model loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);

  const readResult = async () => {
    await page.getByText(/Inference \d+ ms/).waitFor({ timeout: 120000 });
    const type = (await page.locator('p.text-4xl').innerText()).trim();
    const description = (await page.locator('p.text-lg').innerText()).trim();
    const timing = await page.getByText(/Inference \d+ ms/).innerText();
    return { type, description, timing };
  };

  // 1. sample image button
  await page.locator('button[title="Gold hoop earrings"]').click();
  const a = await readResult();
  console.log('sample  ->', a);

  // 2. file upload
  await page.setInputFiles('input[type=file]', imagePath);
  await page.getByRole('button', { name: /Identify Item/i }).click();
  const b = await readResult();
  console.log('upload  ->', b);

  await page.screenshot({ path: 'screenshot.png', fullPage: true });
  await browser.close();

  for (const r of [a, b]) {
    if (!['Earring', 'Necklace'].includes(r.type)) throw new Error(`unexpected type: ${r.type}`);
    if (r.description.split(/\s+/).length < 3) throw new Error(`description too short: ${r.description}`);
  }
  if (errors.length) console.log('browser console errors:\n ' + errors.join('\n '));
  console.log('END-TO-END TEST PASSED');
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
