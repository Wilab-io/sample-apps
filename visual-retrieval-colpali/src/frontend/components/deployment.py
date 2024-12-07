from fasthtml.components import Div, H2, P, Button
from lucide_fasthtml import Lucide

def DeploymentModal():
    return Div(
        Div(
            Div(
                H2(
                    "Application deployment in process",
                    cls="text-xl font-semibold mb-2 text-gray-900 dark:text-white"
                ),
                P(
                    "This process may take multiple minutes",
                    cls="text-gray-500 dark:text-gray-400"
                ),
                Div(
                    Lucide(icon="loader-circle", cls="size-10 animate-spin"),
                    cls="mt-6 flex justify-center"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center"
            ),
            cls="fixed inset-0 flex items-center justify-center z-50 p-4"
        ),
        cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40"
    )

def DeploymentSuccessModal():
    return Div(
        Div(
            Div(
                Div(
                    Lucide(icon="circle-check", cls="size-12 text-green-500"),
                    cls="flex justify-center mb-4"
                ),
                H2(
                    "Deployment successful",
                    cls="text-xl font-semibold mb-2 text-gray-900 dark:text-white"
                ),
                P(
                    "You can start using the search bar",
                    cls="text-gray-500 dark:text-gray-400 mb-6"
                ),
                Button(
                    "OK",
                    cls="w-full p-4 bg-black text-white rounded-[10px] hover:bg-gray-800 transition-colors",
                    onclick="document.getElementById('deployment-modal').remove()"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center"
            ),
            cls="fixed inset-0 flex items-center justify-center z-50 p-4"
        ),
        cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40"
    )

def DeploymentErrorModal():
    return Div(
        Div(
            Div(
                Div(
                    Lucide(icon="circle-x", cls="size-12 text-red-500"),
                    cls="flex justify-center mb-4"
                ),
                H2(
                    "Deployment failed",
                    cls="text-xl font-semibold mb-2 text-gray-900 dark:text-white"
                ),
                P(
                    "The application deployment failed, check the console logs for more details",
                    cls="text-gray-500 dark:text-gray-400 mb-6"
                ),
                Button(
                    "OK",
                    cls="w-full p-4 bg-black text-white rounded-[10px] hover:bg-gray-800 transition-colors",
                    onclick="document.getElementById('deployment-modal').remove()"
                ),
                cls="bg-white dark:bg-gray-900 p-8 rounded-[10px] shadow-md max-w-md w-full text-center"
            ),
            cls="fixed inset-0 flex items-center justify-center z-50 p-4"
        ),
        cls="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm z-40"
    )
