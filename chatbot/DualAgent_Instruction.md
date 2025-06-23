# Dual AI Agent Integration Guide

Congratulations on successfully configuring your business with our AI platform.  
You now have access to two dedicated API endpoints designed to serve both administrative and customer-facing needs.

---

## 1. Business Owner Agent 🏢

**API URL:**  
`http://127.0.0.1:8000/chat/admin/{company_name}/{uid}/`

**Capabilities:**
- Retrieve your complete business data via flexible queries.
- Update or add new information to your business database.

**Usage Instructions:**  
- **Query your business information:**
    ```json
    {
      "action": "query",
      "query": "Show me all my yoghurt item prices"
    }
    ```
- **Update your business information:**
    ```json
    {
      "action": "update",
      "field": "price_list",
      "value": "Updated price list for 2025"
    }
    ```
- **Sample Responses:**  
    - For queries:  
      ```json
      { "answer": "...", "sources": [...] }
      ```
    - For updates:  
      ```json
      { "message": "Field '...' updated with value: ..." }
      ```

---

## 2. Customer Agent 👥

**API URL:**  
`http://127.0.0.1:8000/chat/client/{company_name}/{uid}/`

**Capabilities:**
- Retrieve information such as prices, offers, product lists, FAQs, and more.  
- No permission to update or modify any business data.

**Usage Instructions:**  
- **Ask a question (POST request):**
    ```json
    {
      "query": "Do you have yoghurt items and prices?"
    }
    ```
- **Sample Response:**  
    ```json
    {
      "answer": "Yes, the provided context includes several yoghurt items and their prices. Here they are: ..."
    }
    ```

---

## API Usage Guidelines ⚠️

- **Business Owner Agent**: Retrieve and update access.
- **Customer Agent**: Retrieve-only access.
- **Security Notice:** Each API endpoint is unique to your business and protected by your unique UID. Please keep your admin URL confidential.

---

## Getting Started

1. Save your unique URLs as provided above.
2. Test your agents using Postman or any REST client:
    - For business administration: Test retrieving and updating your business data.
    - For customers: Test typical customer questions.
3. Share the customer agent URL with your clients (for example, on your website or via a QR code).
4. Manage your business data at any time using the business owner agent API.

---

## Need Assistance?

If you have any questions or require assistance with customization, please contact our support team at **011-XXXXXXX**.  
We are here to help.

---

## Thank You

Thank you for choosing our platform to enhance your business capabilities with AI-powered agents.  
We look forward to supporting your continued success.