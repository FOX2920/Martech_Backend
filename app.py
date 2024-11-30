from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import pathlib
import textwrap
import google.generativeai as genai

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_restx import Api
app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='Product Recommendation API', description='API cho hệ thống gợi ý sản phẩm', doc='/swagger')
def find_related_items(target_product_id, top_n=6):

    text_columns=['descript']
    product_engine= "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f"
    query="SELECT * FROM product_data"
    df = pd.read_sql(query, con=product_engine)
    # Kiểm tra sản phẩm mục tiêu
    if target_product_id not in df['product_id'].values:
        raise ValueError(f"Không tìm thấy sản phẩm có ID: {target_product_id}")
    
    # Kết hợp các cột văn bản
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Sử dụng TF-IDF để trích xuất đặc trưng văn bản
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    # Lấy vector của sản phẩm mục tiêu
    target_index = df[df['product_id'] == target_product_id].index[0]
    target_vector = tfidf_matrix[target_index]
    
    # Tính độ tương đồng cosine
    cosine_similarities = cosine_similarity(target_vector, tfidf_matrix)[0]
    
    # Loại bỏ sản phẩm mục tiêu và sắp xếp
    similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
    
    # Trả về các sản phẩm liên quan
    return df.iloc[similar_indices]

def show_product_information(id_pro):
    import pathlib
    import textwrap

    import google.generativeai as genai
    import pandas as pd
    import psycopg2
    from psycopg2 import sql
    from sqlalchemy import create_engine, text
    GOOGLE_API_KEY = "AIzaSyDsql1chYKQ0mI68UyhYbgBIfSK2czni3Y"
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    product_engine= "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f"
    query="SELECT * FROM product_data WHERE product_id = '"+ str(id_pro)+"'"
    pd_df = pd.read_sql(query, con=product_engine)
    response = model.generate_content('bạn có thông tin sản phẩm như sau: '+ str(pd_df.to_string())+ ' hãy miêu tả sinh động sản phẩm cho hấp dẫn người mua, thêm thắt thông tin miêu tả. chỉ trả về kết quả, không giải thích, không ký tự đặc biệt chỉ có dấu . và ,')
    pd_df['miêu tả sản phẩm']=str(response.text)
    return pd_df
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

@app.route('/api/related-items/<string:product_id>', methods=['GET'])
def get_related_items(product_id):
    """
    API to find related items for a given product based on text similarity
    """
    try:
        # Number of related items (optional query parameter, default 6)
        top_n = int(request.args.get('top_n', 6))
        
        # Find related items
        related_items_df = find_related_items(product_id, top_n)
        
        # Convert to list of dictionaries for JSON response
        related_items = related_items_df.to_dict(orient='records')
        
        return jsonify({
            "product_id": product_id, 
            "related_items": related_items
        }), 200
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/product-description/<string:product_id>', methods=['GET'])
def generate_product_description(product_id):
    """
    API to generate an appealing description for a product
    """
    try:
        # Generate product description
        product_info = show_product_information(product_id)
        
        if product_info.empty:
            return jsonify({"error": "Product not found with this ID."}), 404
        
        # Convert result to JSON
        product_data = product_info.to_dict(orient='records')
        
        return jsonify({
            "product_id": product_id, 
            "product_description": product_data[0]['miêu tả sản phẩm']
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
