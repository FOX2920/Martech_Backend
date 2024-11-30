from flask import Flask, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from sqlalchemy import create_engine
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)
api = Api(
    app,
    version='1.0',
    title='Martech API',
    description='API cho Martech',
    doc='/swagger',
    default='Martech',
    default_label='Martech APIs',
    validate=True,
    ordered=True
)

# Database Configuration
DATABASE_CONFIG = {
    'EVENT_URI': "postgresql://ptt_db1_user:cM056SikCQhRErxbPOCP6qTXJTnjWsc7@dpg-ct4joh23esus73ffski0-a.oregon-postgres.render.com/ptt_db1",
    'LOG_URI': "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f",
    'PRODUCT_URI': "postgresql://martech101_fr0f_user:W2FqrFKG6zld1oCd2F3sMs0dYwly0enu@dpg-ct2gvkdsvqrc73aie8i0-a.oregon-postgres.render.com/martech101_fr0f"
}

# Create database engines
event_engine = create_engine(DATABASE_CONFIG['EVENT_URI'])
log_engine = create_engine(DATABASE_CONFIG['LOG_URI'])
product_engine = create_engine(DATABASE_CONFIG['PRODUCT_URI'])

# Configure Google AI
GOOGLE_API_KEY = "AIzaSyDsql1chYKQ0mI68UyhYbgBIfSK2czni3Y"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Helper functions remain the same
def find_related_items(target_product_id, top_n=6):
    """Find related products based on text similarity."""
    text_columns = ['descript']
    query = "SELECT * FROM product_data"
    df = pd.read_sql(query, con=product_engine)
    
    if target_product_id not in df['product_id'].values:
        raise ValueError(f"Không tìm thấy sản phẩm có ID: {target_product_id}")
    
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    target_index = df[df['product_id'] == target_product_id].index[0]
    target_vector = tfidf_matrix[target_index]
    cosine_similarities = cosine_similarity(target_vector, tfidf_matrix)[0]
    
    similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
    return df.iloc[similar_indices]

def show_product_information(id_pro):
    """Generate product description using Gemini AI."""
    query = f"SELECT * FROM product_data WHERE product_id = '{id_pro}'"
    pd_df = pd.read_sql(query, con=product_engine)
    
    response = model.generate_content(
        'bạn có thông tin sản phẩm như sau: ' + 
        str(pd_df.to_string()) + 
        ' hãy miêu tả sinh động sản phẩm cho hấp dẫn người mua, thêm thắt thông tin miêu tả. chỉ trả về kết quả, không giải thích, không ký tự đặc biệt chỉ có dấu . và ,'
    )
    pd_df['miêu tả sản phẩm'] = str(response.text)
    return pd_df

def show_product(id_cus, n_top=10):
    """Generate product recommendations for a customer."""
    try:
        query = f"SELECT * FROM log_recommand WHERE customer_id = '{id_cus}'"
        log_df = pd.read_sql(query, con=log_engine)
        
        if not log_df.empty:
            product_ids_tuple = tuple(log_df['id_product'])
            query2 = f"SELECT * FROM product_data WHERE product_id IN {product_ids_tuple}"
            recommend_product_df = pd.read_sql(query2, con=product_engine)
        else:
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
        
        return recommend_product_df.drop_duplicates(subset=['product_id'])
    except Exception as e:
        print(f"Error in show_product: {e}")
        return pd.DataFrame()

# API Routes using Flask-RESTX
@api.route('/api/product/<string:product_id>')
class Product(Resource):
    @api.doc('get_product',
             params={'product_id': 'The ID of the product'},
             responses={
                 200: 'Success',
                 404: 'Product not found',
                 500: 'Internal server error'
             })
    def get(self, product_id):
        """Get product information by ID"""
        try:
            query = f"SELECT * FROM product_data WHERE product_id = '{product_id}'"
            df = pd.read_sql(query, con=product_engine)
            
            if df.empty:
                return {"error": "Product not found with this ID."}, 404
                
            return {"product": df.to_dict(orient='records')[0]}, 200
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/recommendations/<string:customer_id>')
class Recommendations(Resource):
    @api.doc('get_recommendations',
             params={
                 'customer_id': 'The ID of the customer',
                 'n_top': 'Number of recommendations to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No recommendations found',
                 500: 'Internal server error'
             })
    def get(self, customer_id):
        """Get product recommendations for a customer"""
        try:
            n_top = int(api.payload.get('n_top', 10)) if api.payload else 10
            recommend_product_df = show_product(customer_id, n_top)
            
            if recommend_product_df.empty:
                return {"error": "No recommendations found"}, 404
                
            return {
                "customer_id": customer_id,
                "recommendations": recommend_product_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/related-items/<string:product_id>')
class RelatedItems(Resource):
    @api.doc('get_related_items',
             params={
                 'product_id': 'The ID of the product',
                 'top_n': 'Number of related items to return (default: 6)'
             },
             responses={
                 200: 'Success',
                 404: 'Product not found',
                 500: 'Internal server error'
             })
    def get(self, product_id):
        """Get related items for a product"""
        try:
            top_n = int(api.payload.get('top_n', 6)) if api.payload else 6
            related_items_df = find_related_items(product_id, top_n)
            
            return {
                "product_id": product_id,
                "related_items": related_items_df.to_dict(orient='records')
            }, 200
        except ValueError as ve:
            return {"error": str(ve)}, 404
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/product-description/<string:product_id>')
class ProductDescription(Resource):
    @api.doc('generate_product_description',
             params={'product_id': 'The ID of the product'},
             responses={
                 200: 'Success',
                 404: 'Product not found',
                 500: 'Internal server error'
             })
    def get(self, product_id):
        """Generate AI-powered product description"""
        try:
            product_info = show_product_information(product_id)
            
            if product_info.empty:
                return {"error": "Product not found with this ID."}, 404
                
            return {
                "product_id": product_id,
                "product_description": product_info.to_dict(orient='records')[0]['miêu tả sản phẩm']
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
