# Login Page (`templates/login.html`)

## Purpose
Authenticate users via username or +91 phone (10-digit normalized) plus password.

## Behavior
- Client-side validation: username (>=4 chars) or 10-digit phone (auto +91 prefix), password length >=8.
- Form posts to `/login`; server authenticates via username or phone.
- No OTP required (login kept simple).

## UI
- Inputs: identifier (username/phone), password.
- Messages area for flash feedback.
- CTA card linking to register and home.

## Rationale
- Minimal friction login while enforcing phone format consistency.
- Client-side checks to reduce bad submissions; server enforces real auth.

