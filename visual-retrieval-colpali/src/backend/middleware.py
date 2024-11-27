from fasthtml.common import RedirectResponse
from functools import wraps

def login_required(route_handler):
    @wraps(route_handler)
    async def wrapper(request, *args, **kwargs):
        if isinstance(request, dict):
            session = request
        else:
            session = request.session

        if "user_id" not in session:
            return RedirectResponse("/login", status_code=303)
        return await route_handler(request, *args, **kwargs)
    return wrapper
