# Alpaca API Endpoint Reference

## CORRECT Endpoints (Working):
- Trading API: https://paper-api.alpaca.markets/v2/
- Data API Options: https://data.alpaca.markets/v1beta1/options/
- Account: GET /v2/account
- Orders: POST /v2/orders
- Positions: GET /v2/positions
- Assets: GET /v2/assets/{symbol}

## INCORRECT Endpoints (404 Errors):
- https://data.alpaca.markets/v2/options/ (404)
- https://data.alpaca.markets/v1/options/ (404)
- Double /v2/v2/ in URLs

## Options Trading Notes:
- Use v1beta1 for options data API
- Standard trading API for options orders (v2)
- Option symbols must follow OCC format: SPY240920C00550000

## Paper Trading URLs:
- Base: https://paper-api.alpaca.markets
- API Version: v2 (added automatically by client)

## 404 Error Fix:
The 404 "endpoint not found" error was caused by attempting to access non-existent options endpoints.
Use v1beta1 for options data, not v1 or v2.