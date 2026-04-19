# Marketing — validate demand before writing more code

This folder is the minimum viable launch kit for Agent Kit. If 5–10
pre-orders come in within a week of posting, build more. If crickets,
the product-market fit is wrong and saving yourself a month.

## What's in here

- `index.html` — single-file landing page, no build step, drop on any
  static host (Vercel, Netlify, GitHub Pages, Cloudflare Pages, S3).
- `launch/hn.md` — Show HN post draft + posting notes.
- `launch/reddit.md` — r/SideProject + r/LocalLLaMA drafts.
- `launch/twitter.md` — 8-tweet launch thread.

## Placeholders you need to replace

Every launch file has `{{MUSTACHE}}`-style placeholders. Do a
project-wide find-and-replace before you ship:

| Placeholder            | What it is                              | Example                           |
|------------------------|------------------------------------------|-----------------------------------|
| `{{LANDING_URL}}`      | Public URL of `index.html`               | `https://agentkit.example.com`    |
| `{{BUY_URL}}`          | Gumroad / Lemon Squeezy / Stripe link    | `https://gum.co/agent-kit`        |
| `{{TEAM_BUY_URL}}`     | Team-tier checkout link                  | `https://gum.co/agent-kit-team`   |
| `{{CONTACT_EMAIL}}`    | Your support email                       | `you@example.com`                 |
| `{{MAINTAINER_NAME}}`  | Your real name / handle                  | `Ada Lovelace`                    |
| `{{TWITTER_URL}}`      | Your Twitter / X profile URL             | `https://x.com/adalovelace`       |
| `{{HANDLE}}`           | Your Twitter @handle                     | `@adalovelace`                    |
| `{{NUMBER}}`           | Live pre-order count (day 3+ threads)    | `23`                              |

## 7-day validation plan

This is a **validation** run, not a product launch. The goal is to know
whether people will pay, not to go viral.

**Day 0 — setup (2–3 hours)**
1. Replace all `{{…}}` placeholders in `index.html` and `launch/*.md`.
2. Deploy `index.html`:
   - Vercel: drag-and-drop at vercel.com/new, done in 60 seconds.
   - GitHub Pages: enable Pages on this repo, point to `/marketing`.
3. Set up the checkout:
   - Gumroad is fastest (no review, 10-min setup). Lemon Squeezy
     handles VAT better for EU buyers.
   - Create two products: `$99 Solo`, `$399 Team`. Deliver: a secret
     GitHub repo invite + a Discord invite link.
4. Spin up a free Discord server with `#announcements`, `#support`,
   `#show-and-tell`. Bot it with a role-on-purchase via Gumroad's
   Discord integration.

**Day 1 — soft launch**
- Post to Twitter using `launch/twitter.md`. Best window: Tue–Thu
  9–11am in your buyer's TZ (US-Pacific for dev products).
- DM the link to 10 people who'd have opinions (not a favour ask,
  just "what do you think of this?").
- Post in 2–3 relevant Discord servers' `#self-promo` channels.

**Day 2–3 — amplify**
- Show HN post using `launch/hn.md`. Once only. Tuesday 8am PT ideal.
- r/SideProject on self-promo day (varies by sub rules — check
  sidebar).
- Reply to every comment within 2 hours for first 6 hours.

**Day 4–5 — iterate**
- If you have ≥5 sales: double down. Write a dev-log blog post.
  Reach out to newsletter owners (Python Weekly, Latent Space, etc.)
  with a "behind the scenes" angle.
- If <5 sales: **read the objections**, not the praise. The signal is
  in "I would buy if…" / "I don't get what this does differently
  from…" — not in "cool, bookmarked."

**Day 6–7 — decide**
- Sales ≥ 10 → product is real, start on roadmap (more providers,
  integration test, video walkthrough, affiliate program).
- Sales 3–9 → product is plausible, fix the highest-signal
  objection and try a second launch wave in 3 weeks.
- Sales < 3 → pivot. Either the framing is wrong (rewrite landing
  page + re-launch) or the product is wrong (go ask the 3 buyers
  why they bought and double down on *their* use case).

## What NOT to do during validation

- **Don't add features.** Nobody is ignoring you because you're
  missing Groq support. They're ignoring you because of positioning.
- **Don't drop the price.** $99 filters for serious buyers who'll
  give you useful feedback. Free users give noise.
- **Don't write a blog post about "why I built this" until there's
  evidence someone wants it.** You'll be tempted. Resist.
- **Don't take the first harsh comment personally.** HN will tell
  you evals aren't a moat, that Pydantic is overengineered, and that
  you should have used Rust. Extract the signal; ignore the rest.

## Success metric (be honest with yourself)

The only metric that matters this week: **paid conversions** from the
landing page. Not stars, not likes, not "looks cool" DMs.

- 10+ in 7 days → keep building.
- 3–9 in 7 days → iterate.
- 0–2 in 7 days → the product isn't this. Ask the crickets why.
