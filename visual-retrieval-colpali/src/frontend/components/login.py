from fasthtml.components import Div, H1, P, Form, Input, Button

def Login(error_message=None):
    return Div(
        Div(
            H1(
                "Account login",
                cls="text-4xl font-bold mb-2"
            ),
            P(
                "Welcome! Enter your credentials to use this platform",
                cls="text-gray-600 dark:text-gray-400 mb-8"
            ),
            Form(
                Input(
                    type="text",
                    placeholder="Username",
                    name="username",
                    cls="w-full p-4 mb-4 border rounded-[10px] focus:outline-none focus:border-black dark:bg-gray-800 dark:border-gray-700"
                ),
                Input(
                    type="password",
                    placeholder="Password",
                    name="password",
                    cls="w-full p-4 mb-6 border rounded-[10px] focus:outline-none focus:border-black dark:bg-gray-800 dark:border-gray-700"
                ),
                P(
                    error_message,
                    cls="text-destructive text-sm mb-4 text-center font-medium"
                ) if error_message else None,
                Button(
                    "Login",
                    type="submit",
                    cls="w-full p-4 bg-black text-white rounded-[10px] hover:bg-gray-800 transition-colors"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-lg w-full max-w-md",
                hx_post="/api/login",
                hx_target="#login-form",
                hx_swap="outerHTML",
                hx_trigger="submit",
                hx_indicator="#login-form",
                hx_redirect="/"
            ),
            cls="flex flex-col items-center justify-center h-[calc(100vh-55px)] w-full max-w-screen-xl mx-auto px-4"
        ),
        cls="w-full bg-gray-50 dark:bg-gray-950 min-h-0",
        id="login-form"
    )
