from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_mysqldb import MySQL
import bcrypt
import re

auth_bp = Blueprint('auth', __name__)

# Validasi email format
def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email)

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Cek password
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Validasi input
        if not username or not email or not password or not confirm_password:
            flash("Semua kolom wajib diisi!", "danger")
            return redirect(url_for('auth.register'))

        if not is_valid_email(email):
            flash("Email tidak valid!", "warning")
            return redirect(url_for('auth.register'))

        if password != confirm_password:
            flash("Password tidak cocok!", "warning")
            return redirect(url_for('auth.register'))

        if len(password) < 8:
            flash("Password minimal 8 karakter!", "warning")
            return redirect(url_for('auth.register'))
        # Cek apakah username atau email sudah ada
        mysql = MySQL()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cur.fetchone()

        if existing_user:
            flash("Username atau Email sudah digunakan!", "warning")
            return redirect(url_for('auth.register'))

        # Simpan ke database
        hashed_pw = hash_password(password)
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                    (username, email, hashed_pw))
        mysql.connection.commit()
        cur.close()

        flash("Registrasi berhasil! Silakan login.", "success")
        return redirect(url_for('auth.login'))
        print("Redirect ke login...")
        print("Flash message:", get_flashed_messages())

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        identifier = request.form['identifier']  # bisa username atau email
        password = request.form['password']

        mysql = MySQL()
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s OR email = %s", (identifier, identifier))
        user = cur.fetchone()
        cur.close()

        if user and check_password(password, user[3]):  # user[3] adalah password_hash
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash("Login berhasil!")
            return redirect(url_for('Index'))
        else:
            flash("Username/email atau password salah!")
            return redirect(url_for('auth.login'))

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("Anda telah keluar.")
    return redirect(url_for('auth.login'))