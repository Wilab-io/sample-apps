from fasthtml.components import Div, H1, P, Form, Input, Button

def Login():
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
                    cls="w-full p-4 mb-4 border rounded-[20px] focus:outline-none focus:border-black dark:bg-gray-800 dark:border-gray-700"
                ),
                Input(
                    type="password",
                    placeholder="Password",
                    name="password",
                    cls="w-full p-4 mb-6 border rounded-[20px] focus:outline-none focus:border-black dark:bg-gray-800 dark:border-gray-700"
                ),
                Button(
                    "Login",
                    type="submit",
                    cls="w-full p-4 bg-black text-white rounded-[20px] hover:bg-gray-800 transition-colors"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[20px] shadow-lg w-full max-w-md"
            ),
            cls="flex flex-col items-center justify-center min-h-screen w-full max-w-screen-xl mx-auto px-4"
        ),
        cls="w-full h-full bg-gray-50 dark:bg-gray-950"
    )
