# TODO - Fix Environment and Runtime Issues for e_waste_assistant

## 1. Set GEMINI_API_KEY Environment Variable
- Currently, GEMINI_API_KEY is missing or set to a dummy value.
- This causes chatbot and Gemini price estimation features to fallback with error messages.
- Action:
  - Obtain a valid API key from the Gemini API provider.
  - Set it in your environment variables or in a `.env` file as:
    ```
    GEMINI_API_KEY=your_real_api_key_here
    ```
  - Restart the app after setting the key.

## 2. Device Classification Model File
- The machine learning model file `device_model.pth` does not exist in the project.
- Without it, device image classification falls back to filename-based detection (less accurate).
- Actions:
  - Train the model by running:
    ```
    python model.py
    ```
  - This will generate `device_model.pth` after training.
  - Place the generated `device_model.pth` in the project root (`c:/major project/e_waste_assistant/`).
  - Alternatively, if a pretrained model file is available, add it to the root folder.

## 3. Install Required Python Dependencies
- Ensure the following packages are installed in your Python environment:
  - flask
  - torch
  - torchvision
  - google-generativeai
- You can install them using pip:
  ```
  pip install flask torch torchvision google-generativeai
  ```

## 4. Additional Recommendations
- Verify database `ewaste_assistant.db` initializes correctly on first app run.
- For development, keep Flask debug mode on; switch to production server for deployment.
- Monitor terminal logs for any runtime errors during app usage.
- Test all app features: upload, recommendations, chatbot, admin feedback, etc.

---

# Summary
Fixing these environment configuration and setup issues will resolve most runtime problems and enable full functionality of the app including AI chatbot and device image classification.

If you need, I can assist with creating code to improve runtime messaging or adjust environment loading.

---
