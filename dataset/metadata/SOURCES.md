# Indonesian Adult Content Source List

This document curates the most effective sources for modern Indonesian narrative text, specifically focused on adult entertainment. It also identifies "pollution" targets that should be blacklisted to maintain dataset quality.

## 1. Top-Tier Narrative Sources (High Quality)

These sources consistently provide high volumes of modern, conversational erotic prose.

### **Wattpad Indonesia**
*   **Target Tags:** `#dewasa`, `#21plus`, `#gairah`, `#tabu`, `#dosen`, `#selingkuh`, `#wikwik`.
*   **Strategy:** Requires "infinite scroll" handling. Scroll 10-20 times to populate the chapter lists.
*   **URLs:**
    *   `https://www.wattpad.com/stories/ceritadewasa`
    *   `https://www.wattpad.com/stories/21plus`
    *   `https://www.wattpad.com/stories/birahi`

### **Blogspot & WordPress Ring**
*   **Search Dorks:**
    *   `site:blogspot.com "cerita seks" "kisah nyata" -judi -bola`
    *   `site:wordpress.com "cerita dewasa" "21+"`
*   **Strategy:** Use DuckDuckGo or Google to seed the manifest with these dorks.

## 2. Classic Aggregators (High Volume, Ad Risk)

These sites contain thousands of stories from the 2010-2020 era. They often have aggressive ad-tech and pagination.

*   `ceritadewasa.net` (Classic repository)
*   `cerisex.org`
*   `17tahun.com` (Archive quality)
*   `ceritasex.top`
*   `indosex.net`
*   **Note:** Use "Next Page" logic to ensure full story extraction across multi-page articles.

## 3. The "Pollution" Blacklist (DO NOT SCRAPE)

The following domains often appear in results for "Cerita" (Story) or "Dewasa" (Adult) but contain clean, educational, or irrelevant content.

*   `kompas.id` (News/Literature)
*   `brainacademy.id` (Education)
*   `ruangguru.com` (Education)
*   `halodoc.com` (Health/Medical)
*   `vidio.com` (Video streaming)
*   `goodnovel.com` (Paywalled/Locked content)

## 4. Scraping Best Practices for ID-Landscape

1.  **VPN/DNS:** Many aggregators are blocked by Indonesian ISPs. Use Cloudflare DNS (1.1.1.1) or a VPN.
2.  **Ad-Filtering:** The Language Guard in `scripts/clean_dataset.py` is essential. It filters for "Slot Gacor" and other gambling pollution common in these domains.
3.  **PDFs:** High-quality novels are often stored as PDFs on personal blogs. Use a PDF-to-Text converter for these assets.
