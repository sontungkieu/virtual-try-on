import path from "node:path";
import { expect, test } from "@playwright/test";

const backendUrl = process.env.PLAYWRIGHT_API_BASE_URL ?? "http://127.0.0.1:8000";
const sampleDir = path.resolve(process.cwd(), "../data/eval_set/sample_001");

test("uploads a sample and renders completed artifacts", async ({ page, request }) => {
  const health = await request.get(`${backendUrl}/health`).catch(() => undefined);
  expect(health?.ok(), `Backend is offline or unhealthy at ${backendUrl}`).toBeTruthy();

  await page.goto("/");
  await page.getByLabel("Person image").setInputFiles(path.join(sampleDir, "person.jpg"));
  await page.getByText("Top", { exact: true }).first().click();
  await page.getByLabel("Top garment image").setInputFiles(path.join(sampleDir, "garment_top.jpg"));
  await page.getByLabel("FLUX refine").uncheck();
  await page.getByLabel("Repair").uncheck();
  await page.getByRole("button", { name: "Generate" }).click();

  await expect(page.getByTestId("tryon-result")).toBeVisible({ timeout: 15 * 60 * 1000 });
  await expect(page.getByRole("region", { name: "Quality report" })).toBeVisible();
  await expect(page.getByText("Artifacts", { exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("Traceback (most recent call last)");
});
