# Register Page (`templates/register.html`)

## Purpose
Create new accounts with full name, +91 phone, username, and strong password.

## Validation
- Client-side: name >=3, phone matches +91XXXXXXXXXX, username >=4, password regex (upper/lower/number/special, len>=8).
- Server-side (`/register` in `app.py`): same checks plus uniqueness for username and phone; passwords stored hashed.

## UI
- Inputs: full name, phone, username, password.
- Helper text for phone format and password strength.
- Flash messages area for errors/success.

## Rationale
- Enforces unique identity (username + phone).
- Strong password policy; consistent +91 formatting for later contact/logging.

