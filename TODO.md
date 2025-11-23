# Task: Implement device recycling price estimation with Gemini model and nearest recycling center suggestion

## Steps:

1. Database Updates
   - Add columns `estimated_price` (REAL) and `user_address` (TEXT) to device_submissions table.
   - Update database.py with methods to save and retrieve these new fields.

2. Model Integration
   - Extend model.py with a new function `estimate_recycling_price` to use the Gemini fine-tuned model.
   - This function will take device info and user inputs and return price estimate.

3. Backend Route Changes (app.py)
   - Modify `/questions` page handler to include address input form.
   - Modify `/recommendations` POST handler to:
     - Call `estimate_recycling_price` with device info and user inputs.
     - Find nearest recycling centers based on user address.
     - Store estimated price and address in device_submissions.
     - Pass estimated price to recommendations template.

4. Frontend Template Changes (templates/index.html)
   - Update questions page to add address input fields.
   - Update recommendations page to display estimated recycling price prominently.
   - Potential UI improvements around upload and address inputs.

5. Additional Enhancements
   - Implement simple city or locality matching logic for nearest recycling centers.
   - Add input validation and error messages for address fields.
   - Maintain user session validation on these routes.

## Notes:
- All features will be behind login protection.
- Recycling centers data is already available in database for suggestions.
- Implementation will start with DB updates and then proceed stepwise.
