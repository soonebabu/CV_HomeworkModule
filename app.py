from flask import Flask, render_template


def create_app():
    app = Flask(__name__)
    app.secret_key = "csc8830-cv-dev-key"  # fine for a local class demo, not for production

    from homeworks.hw2_calibration.routes import bp as hw2_bp
    from homeworks.hw3_blurring.routes import bp as hw3_bp
    from homeworks.hw4_segmentation.routes import bp as hw4_bp
    from homeworks.hw6_motion.routes import bp as hw6_bp
    from homeworks.samples import bp as samples_bp

    app.register_blueprint(hw2_bp, url_prefix="/hw2")
    app.register_blueprint(hw3_bp, url_prefix="/hw3")
    app.register_blueprint(hw4_bp, url_prefix="/hw4")
    app.register_blueprint(hw6_bp, url_prefix="/hw6")
    app.register_blueprint(samples_bp, url_prefix="/samples")

    @app.route("/")
    def landing():
        return render_template("landing.html")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
