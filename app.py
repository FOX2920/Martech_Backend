from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from sqlalchemy import create_engine, text
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime
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
    'LOG_URI': "postgresql://ptt_db3_user:HNGNAc9xuyZAByLyJUkGqdM0QJbgOXBz@dpg-ct60msbv2p9s7394d94g-a.oregon-postgres.render.com/ptt_db3",
    'PRODUCT_URI': "postgresql://ptt_db3_user:HNGNAc9xuyZAByLyJUkGqdM0QJbgOXBz@dpg-ct60msbv2p9s7394d94g-a.oregon-postgres.render.com/ptt_db3",
    'ORDER_URI': "postgresql://ptt_db2_user:jCcmpSkXRsKUYWr50uXRwTkvkmD9LVro@dpg-ct4jp4aj1k6c73egvbq0-a.oregon-postgres.render.com/ptt_db2",
    'CART_URI': "postgresql://ptt_db2_user:jCcmpSkXRsKUYWr50uXRwTkvkmD9LVro@dpg-ct4jp4aj1k6c73egvbq0-a.oregon-postgres.render.com/ptt_db2"
}

# Create database engines
event_engine = create_engine(DATABASE_CONFIG['EVENT_URI'])
log_engine = create_engine(DATABASE_CONFIG['LOG_URI'])
product_engine = create_engine(DATABASE_CONFIG['PRODUCT_URI'])
order_engine = create_engine(DATABASE_CONFIG['ORDER_URI'])
cart_engine = create_engine(DATABASE_CONFIG['CART_URI'])
# Configure Google AI
GOOGLE_API_KEY = "AIzaSyDsql1chYKQ0mI68UyhYbgBIfSK2czni3Y"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def find_top_n_related_items_by_keyword(keyword, top_n=10):
    # SQL query tìm kiếm top_n sản phẩm
    search_query = """
    SELECT *
    FROM product_data
    WHERE descript ILIKE %(keyword)s
    LIMIT %(top_n)s;
    """

    # Tham số tìm kiếm
    params = {
        'keyword': f"%{keyword}%",
        'top_n': top_n
    }

    # Thực thi truy vấn và đọc kết quả vào DataFrame
    try:
        return pd.read_sql(search_query, product_engine, params=params)
    except Exception as e:
        print(f"Error in find_top_n_related_items_by_keyword: {e}")
        return pd.DataFrame()

def find_top_n_related_items_by_keyword_danhmuc(keyword, danhmuc , top_n=10):
    # SQL query tìm kiếm top_n sản phẩm
    search_query = """
    SELECT *
    FROM product_data
    WHERE descript ILIKE %(keyword)s AND product_category_name = %(danhmuc)s
    LIMIT %(top_n)s;
    """

    # Tham số tìm kiếm
    params = {
        'keyword': f"%{keyword}%",
        'danhmuc': str(danhmuc),
        'top_n': top_n
    }

    # Thực thi truy vấn và đọc kết quả vào DataFrame
    try:
        return pd.read_sql(search_query, product_engine, params=params)
    except Exception as e:
        print(f"Error in find_top_n_related_items_by_keyword_khoang_gia: {e}")
        return None

def find_top_n_related_items_by_keyword_khoang_gia(keyword, min, max, top_n=10):
    # SQL query tìm kiếm top_n sản phẩm
    search_query = """
    SELECT *
    FROM product_data
    WHERE descript ILIKE %(keyword)s AND price >= %(min)s AND price <= %(max)s
    LIMIT %(top_n)s;
    """

    # Tham số tìm kiếm
    params = {
        'keyword': f"%{keyword}%",
        'min': float(min),
        'max': float(max),
        'top_n': top_n
    }

    # Thực thi truy vấn và đọc kết quả vào DataFrame
    try:
        return pd.read_sql(search_query, product_engine, params=params)
    except Exception as e:
        print(f"Error in find_top_n_related_items_by_keyword_khoang_gia: {e}")
        return None
        
# New Helper Functions
def find_top_n_related_items_by_keyword_gia(keyword, gia, top_n=10):
    """Find products by keyword and price."""
    search_query = """
    SELECT *
    FROM product_data
    WHERE descript ILIKE %(keyword)s AND price = %(gia)s
    LIMIT %(top_n)s;
    """
    params = {
        'keyword': f"%{keyword}%",
        'gia': float(gia),
        'top_n': top_n
    }
    try:
        return pd.read_sql(search_query, product_engine, params=params)
    except Exception as e:
        print(f"Error in find_top_n_related_items_by_keyword_gia: {e}")
        return None

def create_cart(customer_id, product_id):
    """
    Create a new cart entry or update existing one.
    Args:
        customer_id (str): Customer ID
        product_id (str): Product ID
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if item already exists in cart
        check_query = """
            SELECT * FROM cart_data 
            WHERE customer_id = %(customer_id)s 
            AND product_id = %(product_id)s
        """
        
        with cart_engine.connect() as conn:
            result = pd.read_sql(
                check_query,
                conn,
                params={
                    "customer_id": str(customer_id),
                    "product_id": str(product_id)
                }
            )
            
            if result.empty:
                # If item doesn't exist, create new entry
                insert_query = """
                    INSERT INTO cart_data (customer_id, product_id)
                    VALUES (%(customer_id)s, %(product_id)s)
                """
                conn.execute(text(insert_query), {
                    "customer_id": str(customer_id),
                    "product_id": str(product_id)
                })
                conn.commit()
                
        return True
        
    except Exception as e:
        print(f"Error in create_cart: {e}")
        return False

def get_cart_items(customer_id):
    """
    Get all cart items for a specific customer.
    Args:
        customer_id (str): Customer ID
    Returns:
        pandas.DataFrame: DataFrame containing cart items with product details
    """
    try:
        query = """
            SELECT c.*, p.*
            FROM cart_data c
            JOIN product_data p ON c.product_id = p.product_id
            WHERE c.customer_id = %(customer_id)s
        """
        
        with cart_engine.connect() as conn:
            result = pd.read_sql(
                query,
                conn,
                params={"customer_id": str(customer_id)}
            )
            
        return result
        
    except Exception as e:
        print(f"Error in get_cart_items: {e}")
        return pd.DataFrame()

def remove_cart_item(customer_id, product_id):
    """
    Remove an item from the cart.
    Args:
        customer_id (str): Customer ID
        product_id (str): Product ID
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        delete_query = """
            DELETE FROM cart_data 
            WHERE customer_id = %(customer_id)s 
            AND product_id = %(product_id)s
        """
        
        with cart_engine.connect() as conn:
            conn.execute(text(delete_query), {
                "customer_id": str(customer_id),
                "product_id": str(product_id)
            })
            conn.commit()
            
        return True
        
    except Exception as e:
        print(f"Error in remove_cart_item: {e}")
        return False

def clear_cart(customer_id):
    """
    Remove all items from a customer's cart.
    Args:
        customer_id (str): Customer ID
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        delete_query = """
            DELETE FROM cart_data 
            WHERE customer_id = %(customer_id)s
        """
        
        with cart_engine.connect() as conn:
            conn.execute(text(delete_query), {
                "customer_id": str(customer_id)
            })
            conn.commit()
            
        return True
        
    except Exception as e:
        print(f"Error in clear_cart: {e}")
        return False

# New helper functions
def record_event(customer_id, product_id, record_type):
    """Record a customer event."""
    data = {
        "customer_id": str(customer_id),
        "id_product": str(product_id),
        "event_type": str(record_type),
        "event_time": str(datetime.datetime.now()),
        "date": str(datetime.date.today())
    }
    df = pd.DataFrame([data])
    df.to_sql("event_data", event_engine, if_exists='append', index=False)
    return True

def show_product_by_danhmuc(danhmuc, top_n=10):
    """Get products by category."""
    search_query = """
    SELECT *
    FROM product_data
    WHERE product_category_name = %(danhmuc)s
    LIMIT %(top_n)s;
    """
    params = {
        'danhmuc': str(danhmuc),
        'top_n': top_n
    }
    try:
        return pd.read_sql(search_query, product_engine, params=params)
    except Exception as e:
        print(f"Error in show_product_by_danhmuc: {e}")
        return pd.DataFrame()

def create_order(order_id, customer_id, order_status, order_purchase_timestamp,
                total_cost, First_Name, LastName, Street_Address, Province,
                City, Zipcode, Phone, Apt_Suite, Email, opp_id='',
                Phuong_thuc_thanh_toan=''):
    """Create a new order."""
    data = {
        'order_id': str(order_id),
        'customer_id': str(customer_id),
        'order_status': str(order_status),
        'order_purchase_timestamp': str(order_purchase_timestamp),
        'total cost': float(total_cost),
        'opp_id': str(opp_id),
        'First_Name': str(First_Name),
        'LastName': str(LastName),
        'Street_Address': str(Street_Address),
        'Province': str(Province),
        'City': str(City),
        'Zipcode': str(Zipcode),
        'Phone': str(Phone),
        'Apt_Suite': str(Apt_Suite),
        'Email': str(Email),
        'Phuong_thuc_thanh_toan': str(Phuong_thuc_thanh_toan)
    }
    df = pd.DataFrame([data])
    df.to_sql("order_data", order_engine, if_exists='append', index=False)
    return True

def create_order_item(order_id, product_id, price, seller_id='', shipping_charges=0):
    """Create a new order item."""
    data = {
        'order_id': str(order_id),
        'product_id': str(product_id),
        'seller_id': str(seller_id),
        'price': str(price),
        'shipping_charges': str(shipping_charges)
    }
    df = pd.DataFrame([data])
    df.to_sql("order_item_data", order_engine, if_exists='append', index=False)
    return True
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
        query = f"SELECT * FROM log_recommand_data WHERE customer_id = '{id_cus}'"
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
            n_top = int(request.args.get('n_top', 10))
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
            top_n = int(request.args.get('top_n', 6))
            related_items_df = find_related_items(product_id, top_n).to_dict(orient='records')
            
            return {
                "product_id": product_id,
                "related_items": related_items_df
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

# New API Routes
@api.route('/api/event')
class EventRecord(Resource):
    @api.doc('record_event',
             params={
                 'customer_id': 'The ID of the customer',
                 'product_id': 'The ID of the product',
                 'record_type': 'The type of event'
             },
             responses={
                 200: 'Success',
                 500: 'Internal server error'
             })
    def post(self):
        """Record a customer event"""
        try:
            data = request.json
            record_event(
                data['customer_id'],
                data['product_id'],
                data['record_type']
            )
            return {"message": "Event recorded successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/products/category/<string:category>')
class ProductsByCategory(Resource):
    @api.doc('get_products_by_category',
             params={
                 'category': 'The category name',
                 'top_n': 'Number of products to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No products found',
                 500: 'Internal server error'
             })
    def get(self, category):
        """Get products by category"""
        try:
            top_n = int(request.args.get('top_n', 10))
            products_df = show_product_by_danhmuc(category, top_n)
            
            if products_df.empty:
                return {"error": "No products found in this category"}, 404
                
            return {
                "category": category,
                "products": products_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500
            
# Define request models
order_model = api.model('Order', {
    'order_id': fields.String(required=True, description='Unique order identifier'),
    'customer_id': fields.String(required=True, description='Customer identifier'),
    'order_status': fields.String(required=True, description='Order status'),
    'order_purchase_timestamp': fields.String(required=True, description='Purchase timestamp'),
    'total_cost': fields.Float(required=True, description='Total cost of the order'),
    'First_Name': fields.String(required=True, description='Customer first name'),
    'LastName': fields.String(required=True, description='Customer last name'),
    'Street_Address': fields.String(required=True, description='Street address'),
    'Province': fields.String(required=True, description='Province'),
    'City': fields.String(required=True, description='City'),
    'Zipcode': fields.String(required=True, description='Zip code'),
    'Phone': fields.String(required=True, description='Phone number'),
    'Apt_Suite': fields.String(required=True, description='Apartment/Suite number'),
    'Email': fields.String(required=True, description='Email address'),
    'opp_id': fields.String(required=False, description='Opportunity ID'),
    'Phuong_thuc_thanh_toan': fields.String(required=False, description='Payment method')
})

order_item_model = api.model('OrderItem', {
    'order_id': fields.String(required=True, description='Order identifier'),
    'product_id': fields.String(required=True, description='Product identifier'),
    'price': fields.Float(required=True, description='Price of the item'),
    'seller_id': fields.String(required=False, description='Seller identifier'),
    'shipping_charges': fields.Float(required=False, default=0, description='Shipping charges')
})

@api.route('/api/orders')
class Orders(Resource):
    @api.doc('create_order',
             responses={
                 200: 'Success',
                 400: 'Invalid request',
                 500: 'Internal server error'
             })
    @api.expect(order_model, validate=True)
    def post(self):
        """Create a new order"""
        try:
            data = request.json
            create_order(
                data['order_id'],
                data['customer_id'],
                data['order_status'],
                data['order_purchase_timestamp'],
                data['total_cost'],
                data['First_Name'],
                data['LastName'],
                data['Street_Address'],
                data['Province'],
                data['City'],
                data['Zipcode'],
                data['Phone'],
                data['Apt_Suite'],
                data['Email'],
                data.get('opp_id', ''),
                data.get('Phuong_thuc_thanh_toan', '')
            )
            return {"message": "Order created successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/order-items')
class OrderItems(Resource):
    @api.doc('create_order_item',
             responses={
                 200: 'Success',
                 400: 'Invalid request',
                 500: 'Internal server error'
             })
    @api.expect(order_item_model, validate=True)
    def post(self):
        """Create a new order item"""
        try:
            data = request.json
            create_order_item(
                data['order_id'],
                data['product_id'],
                data['price'],
                data.get('seller_id', ''),
                data.get('shipping_charges', 0)
            )
            return {"message": "Order item created successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

@api.route('/api/products/search')
class ProductSearch(Resource):
    @api.doc('search_products',
             params={
                 'keyword': 'Search keyword',
                 'price': 'Product price',
                 'top_n': 'Number of products to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No products found',
                 500: 'Internal server error'
             })
    def get(self):
        """Search products by keyword and price"""
        try:
            keyword = request.args.get('keyword', '')
            price = float(request.args.get('price', 0))
            top_n = int(request.args.get('top_n', 10))
            
            products_df = find_top_n_related_items_by_keyword_gia(keyword, price, top_n)
            
            if products_df.empty:
                return {"error": "No products found matching the criteria"}, 404
                
            return {
                "keyword": keyword,
                "price": price,
                "products": products_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500


@api.route('/api/cart')
class Cart(Resource):
    @api.doc('create_cart',
             params={
                 'customer_id': 'The ID of the customer',
                 'product_id': 'The ID of the product'
             },
             responses={
                 200: 'Success',
                 400: 'Bad Request',
                 415: 'Unsupported Media Type',
                 500: 'Internal server error'
             })
    @api.expect(api.model('CartItem', {
        'customer_id': fields.String(required=True, description='Customer identifier'),
        'product_id': fields.String(required=True, description='Product identifier')
    }))
    def post(self):
        """Add item to cart"""
        if not request.is_json:
            return {"error": "Request must be JSON"}, 415
            
        try:
            data = request.get_json()
            success = create_cart(
                data['customer_id'],
                data['product_id']
            )
            
            if success:
                return {"message": "Item added to cart successfully"}, 200
            else:
                return {"error": "Failed to add item to cart"}, 500
                
        except Exception as e:
            return {"error": str(e)}, 500

    @api.doc('get_cart',
             params={'customer_id': 'The ID of the customer'},
             responses={
                 200: 'Success',
                 404: 'Cart not found',
                 500: 'Internal server error'
             })
    def get(self):
        """Get customer's cart items"""
        try:
            customer_id = request.args.get('customer_id')
            if not customer_id:
                return {"error": "customer_id is required"}, 400
                
            cart_df = get_cart_items(customer_id)
            
            if cart_df.empty:
                return {"message": "Cart is empty", "cart_items": []}, 200
                
            return {
                "customer_id": customer_id,
                "cart_items": cart_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500
            
    @api.doc('delete_cart_item',
             params={
                 'customer_id': 'The ID of the customer',
                 'product_id': 'The ID of the product'
             },
             responses={
                 200: 'Success',
                 400: 'Bad Request',
                 500: 'Internal server error'
             })
    def delete(self):
        """Remove item from cart"""
        try:
            customer_id = request.args.get('customer_id')
            product_id = request.args.get('product_id')
            
            if not customer_id or not product_id:
                return {"error": "Both customer_id and product_id are required"}, 400
                
            success = remove_cart_item(customer_id, product_id)
            
            if success:
                return {"message": "Item removed from cart successfully"}, 200
            else:
                return {"error": "Failed to remove item from cart"}, 500
                
        except Exception as e:
            return {"error": str(e)}, 500

# Define the request model for user creation
user_model = api.model('User', {
    'customer_id': fields.String(required=True, description='Unique customer identifier'),
    'customer_zip_code_prefix': fields.String(required=True, description='Customer zip code prefix'),
    'customer_city': fields.String(required=True, description='City of the customer'),
    'customer_state': fields.String(required=True, description='State of the customer'),
    'email': fields.String(required=True, description='Email of the customer'),
    'name': fields.String(required=True, description='Name of the customer'),
    'phone': fields.String(required=True, description='Phone number of the customer'),
    'password': fields.String(required=True, description='Password of the customer'),
    'taikhoan': fields.String(required=True, description='Account information'),
    'khung_gio_vao_web_trung_binh': fields.String(required=False, description='Average time the customer enters the website'),
    'ngay_mua_tiep_theo': fields.String(required=False, description='Date of the next purchase'),
    'opp_id': fields.String(required=False, description='Opportunity ID')
})

@api.route('/api/user')
class User(Resource):
    @api.doc('create_user',
             responses={
                 200: 'User created successfully',
                 400: 'Invalid request',
                 500: 'Internal server error'
             })
    @api.expect(user_model, validate=True)
    def post(self):
        """Create a new user"""
        try:
            data = request.json
            create_user(
                data['customer_id'],
                data['customer_zip_code_prefix'],
                data['customer_city'],
                data['customer_state'],
                data['email'],
                data['name'],
                data['phone'],
                data['password'],
                data['taikhoan'],
                data.get('khung_gio_vao_web_trung_binh', ''),
                data.get('ngay_mua_tiep_theo', ''),
                data.get('opp_id', '')
            )
            return {"message": "User created successfully"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Search Products with Price Range
@api.route('/api/products/search/price-range')
class ProductSearchPriceRange(Resource):
    @api.doc('search_products_price_range',
             params={
                 'keyword': 'Search keyword',
                 'min_price': 'Minimum price',
                 'max_price': 'Maximum price',
                 'top_n': 'Number of products to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No products found',
                 500: 'Internal server error'
             })
    def get(self):
        """Search products by keyword and price range"""
        try:
            keyword = request.args.get('keyword', '')
            min_price = float(request.args.get('min_price', 0))
            max_price = float(request.args.get('max_price', float('inf')))
            top_n = int(request.args.get('top_n', 10))
            
            products_df = find_top_n_related_items_by_keyword_khoang_gia(
                keyword, min_price, max_price, top_n
            )
            
            if products_df is None or products_df.empty:
                return {"error": "No products found matching the criteria"}, 404
                
            return {
                "keyword": keyword,
                "min_price": min_price,
                "max_price": max_price,
                "products": products_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Search Products by Category and Keyword
@api.route('/api/products/search/category')
class ProductSearchCategory(Resource):
    @api.doc('search_products_category',
             params={
                 'keyword': 'Search keyword',
                 'category': 'Product category',
                 'top_n': 'Number of products to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No products found',
                 500: 'Internal server error'
             })
    def get(self):
        """Search products by keyword and category"""
        try:
            keyword = request.args.get('keyword', '')
            category = request.args.get('category', '')
            top_n = int(request.args.get('top_n', 10))
            
            products_df = find_top_n_related_items_by_keyword_danhmuc(
                keyword, category, top_n
            )
            
            if products_df is None or products_df.empty:
                return {"error": "No products found matching the criteria"}, 404
                
            return {
                "keyword": keyword,
                "category": category,
                "products": products_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Simple Keyword Search
@api.route('/api/products/search/keyword')
class ProductSearchKeyword(Resource):
    @api.doc('search_products_keyword',
             params={
                 'keyword': 'Search keyword',
                 'top_n': 'Number of products to return (default: 10)'
             },
             responses={
                 200: 'Success',
                 404: 'No products found',
                 500: 'Internal server error'
             })
    def get(self):
        """Search products by keyword only"""
        try:
            keyword = request.args.get('keyword', '')
            top_n = int(request.args.get('top_n', 10))
            
            products_df = find_top_n_related_items_by_keyword(keyword, top_n)
            
            if products_df.empty:
                return {"error": "No products found matching the criteria"}, 404
                
            return {
                "keyword": keyword,
                "products": products_df.to_dict(orient='records')
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
