from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
import psycopg2

app = Flask(__name__)
CORS(app)

# Database Connection URIs
EVENT_DATABASE_URI = "postgresql://ptt_db1_user:cM056SikCQhRErxbPOCP6qTXJTnjWsc7@dpg-ct4joh23esus73ffski0-a.oregon-postgres.render.com/ptt_db1"
LOG_DATABASE_URI = "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f"
PRODUCT_DATABASE_URI = "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f"

# Create database engines
event_engine = create_engine(EVENT_DATABASE_URI)
log_engine = create_engine(LOG_DATABASE_URI)
product_engine = create_engine(PRODUCT_DATABASE_URI)

def show_product(id_cus, n_top=10):
    """
    Generate product recommendations for a customer
    
    Args:
        id_cus (str): Customer ID
        n_top (int, optional): Number of top products to recommend. Defaults to 10.
    
    Returns:
        pandas.DataFrame: Recommended products
    """
    try:
        # Check if customer has visited the website before
        query = f"SELECT * FROM log_recommand WHERE customer_id = '{id_cus}'"
        log_df = pd.read_sql(query, con=log_engine)
        
        if not log_df.empty:
            print('Customer has visited the website before')
            # Get products based on customer's previous interactions
            product_ids_tuple = tuple(log_df['id_product'])
            query2 = f"SELECT * FROM product_data WHERE product_id IN {product_ids_tuple}"
            recommend_product_df = pd.read_sql(query2, con=product_engine)
        else:
            print('Customer has not visited the website before')
            # Get most popular products if no previous interactions
            query = f"""
            SELECT id_product FROM (
                SELECT id_product, COUNT(*) 
                FROM event_data 
                GROUP BY id_product 
                ORDER BY COUNT(*) DESC
                LIMIT {n_top}
            ) top_products
            """
            recommend_product_df_id = pd.read_sql(query, con=event_engine)
            
            product_ids_tuple = tuple(recommend_product_df_id['id_product'])
            query2 = f"SELECT * FROM product_data WHERE product_id IN {product_ids_tuple}"
            recommend_product_df = pd.read_sql(query2, con=product_engine)
        
        # Remove duplicate products
        recommend_product_df = recommend_product_df.drop_duplicates(subset=['product_id'])
        return recommend_product_df
    
    except Exception as e:
        print(f"Error in show_product: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs

@app.route('/api/product/<string:product_id>', methods=['GET'])
def get_product(product_id):
    """
    API to retrieve product information based on product_id
    """
    try:
        # Query data from PostgreSQL
        query = f"SELECT * FROM product_data WHERE product_id = '{product_id}'"
        df = pd.read_sql(query, con=product_engine)

        if df.empty:
            return jsonify({"error": "Product not found with this ID."}), 404

        # Convert DataFrame to JSON
        product_data = df.to_dict(orient='records')
        return jsonify({"product": product_data[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/<string:customer_id>', methods=['GET'])
def get_recommendations(customer_id):
    """
    API to retrieve product recommendations for a customer
    """
    try:
        # Number of recommendations (optional query parameter, default 10)
        n_top = int(request.args.get('n_top', 10))
        
        # Get recommendations
        recommend_product_df = show_product(customer_id, n_top)
        
        if recommend_product_df.empty:
            return jsonify({"error": "No recommendations found"}), 404
        
        # Convert recommendations to list of dictionaries
        recommendations = recommend_product_df.to_dict(orient='records')
        
        return jsonify({
            "customer_id": customer_id, 
            "recommendations": recommendations
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)