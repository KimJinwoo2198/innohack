set -e

unset SKIP_DOCKER_INIT

echo "Collecting static files..."
python manage.py collectstatic --noinput --clear || {
    echo "collectstatic with --clear failed, trying without --clear..."
    python manage.py collectstatic --noinput || {
        echo "collectstatic failed, continuing without static files..."
        echo "This may be due to S3 configuration issues. Please check your AWS credentials and bucket settings."
    }
}

exec "$@" 