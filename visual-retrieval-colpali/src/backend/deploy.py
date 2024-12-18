import os
from dotenv import load_dotenv
import logging

from vespa.package import (
    ApplicationPackage,
    AuthClient,
    Parameter,
    RankProfile,
)
from vespa.configuration.services import (
    services,
    container,
    search,
    document_api,
    document_processing,
    clients,
    client,
    config,
    content,
    redundancy,
    documents,
    node,
    certificate,
    document,
    nodes,
)
from vespa.configuration.vt import vt
from vespa.package import ServicesConfiguration
from backend.models import UserSettings

import pty
import subprocess
import re
import time
import select

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("vespa_app")

async def deploy_application_step_1(settings: UserSettings):
    logger.info("Validating settings")
    if not all([
        settings.tenant_name,
        settings.app_name,
        settings.instance_name,
        settings.gemini_token
    ]):
        raise ValueError("Missing required settings")

    VESPA_TENANT_NAME = settings.tenant_name
    VESPA_APPLICATION_NAME = settings.app_name
    VESPA_INSTANCE_NAME = settings.instance_name

    logger.info(f"Deploying application {VESPA_APPLICATION_NAME} to tenant {VESPA_TENANT_NAME}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    app_dir = os.path.join(parent_dir, "application")

    current_dir = os.getcwd()

    try:
        os.chdir(app_dir)

        subprocess.run(["vespa", "config", "set", "target", "cloud"], check=True)

        app_config = f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}"
        subprocess.run(["vespa", "config", "set", "application", app_config], check=True)

        subprocess.run(["vespa", "auth", "cert", "-f", "-a", f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}"], check=True)

        def run_auth_login():
            master, slave = pty.openpty()
            process = subprocess.Popen(
                ["vespa", "auth", "login", "-a", f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                text=True,
                cwd=app_dir
            )

            output = ""
            answered_prompt = False

            while True:
                r, _, _ = select.select([master], [], [], 0.1)
                if r:
                    try:
                        data = os.read(master, 1024).decode()
                        output += data

                        if not answered_prompt and "Automatically open confirmation page in your default browser? [Y/n]" in output:
                            os.write(master, "n\n".encode())
                            answered_prompt = True

                        # Once we find the URL, we can return
                        if "Please open link in your browser: " in data:
                            # Kill the process since we don't need to wait for completion
                            process.kill()
                            os.close(master)
                            os.close(slave)
                            return output

                    except OSError:
                        break

                # If process ends before we find URL, that's an error
                if process.poll() is not None:
                    break

            # Clean up
            process.kill()
            os.close(master)
            os.close(slave)
            return output

        # Create and start daemon thread for auth login
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_auth_login)
            output = future.result()

        logger.debug(f"Full output: {output}")

        url_match = re.search(r'Please open link in your browser: (https://[^\s]+)', output)
        if not url_match:
            raise Exception("Could not find authentication URL in command output")

        auth_url = url_match.group(1)
        logger.info(f"Authentication URL found: {auth_url}")

        # Load certificate files
        cert_dir = os.path.expanduser(f"~/.vespa/{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}")
        logger.debug(f"Looking for certificates in: {cert_dir}")

        private_key_path = os.path.join(cert_dir, "data-plane-private-key.pem")
        public_cert_path = os.path.join(cert_dir, "data-plane-public-cert.pem")

        # Wait a bit for files to be created
        max_retries = 10
        retry_count = 0
        while retry_count < max_retries:
            if os.path.exists(private_key_path) and os.path.exists(public_cert_path):
                break
            time.sleep(1)
            retry_count += 1
            logger.debug(f"Waiting for certificate files... attempt {retry_count}/{max_retries}")

        if not os.path.exists(private_key_path) or not os.path.exists(public_cert_path):
            raise FileNotFoundError(f"Certificate files not found in {cert_dir}")

        # Read certificate files
        with open(private_key_path, 'r') as f:
            private_key = f.read()
        with open(public_cert_path, 'r') as f:
            public_cert = f.read()

        # Set environment variables
        os.environ["VESPA_CLOUD_MTLS_KEY"] = private_key
        os.environ["VESPA_CLOUD_MTLS_CERT"] = public_cert

        logger.info("Successfully loaded certificate files")

        return {"status": "success", "auth_url": auth_url}

    except Exception as e:
        raise
    finally:
        os.chdir(current_dir)

async def deploy_application_step_2(request, settings: UserSettings, user_id: str):
    """Deploy the Vespa application"""
    # Load environment variables
    load_dotenv()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    VESPA_TENANT_NAME = settings.tenant_name
    VESPA_APPLICATION_NAME = settings.app_name
    VESPA_INSTANCE_NAME = settings.instance_name

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(base_dir))
    app_dir = os.path.join(parent_dir, "application")

    copy_api_key_file(parent_dir, VESPA_TENANT_NAME)

    try:
        logger.debug("Generating services.xml")

        service_config = ServicesConfiguration(
            application_name=VESPA_APPLICATION_NAME,
            services_config=services(
                container(
                    search(),
                    document_api(),
                    document_processing(),
                    clients(
                        client(
                            certificate(file="security/clients.pem"),
                            id="mtls",
                            permissions="read,write",
                        ),
                    ),
                    config(
                        vt("tag")(
                            vt("bold")(
                                vt("open", "<strong>"),
                                vt("close", "</strong>"),
                            ),
                            vt("separator", "..."),
                        ),
                        name="container.qr-searchers",
                    ),
                    id=f"{VESPA_APPLICATION_NAME}_container",
                    version="1.0",
                ),
                content(
                    redundancy("1"),
                    documents(document(type="pdf_page", mode="index")),
                    nodes(node(distribution_key="0", hostalias="node1")),
                    config(
                        vt("max_matches", "2", replace_underscores=False),
                        vt("length", "1000"),
                        vt("surround_max", "500", replace_underscores=False),
                        vt("min_length", "300", replace_underscores=False),
                        name="vespa.config.search.summary.juniperrc",
                    ),
                    id=f"{VESPA_APPLICATION_NAME}_content",
                    version="1.0",
                ),
                version="1.0",
            ),
        )

        # Create the Vespa application package and save it to services.xml
        vespa_application_package = ApplicationPackage(
            name=VESPA_APPLICATION_NAME,
            services_config=service_config,
            auth_clients=[
                AuthClient(
                    id="mtls",
                    permissions="read,write",
                    parameters=[Parameter("certificate", {"file": "security/clients.pem"})],
                ),
            ],
        )

        servicesXml = vespa_application_package.services_to_text
        services_xml_path = os.path.join(app_dir, "services.xml")
        with open(services_xml_path, "w") as f:
            f.write(servicesXml)

        # Write the schema to its file
        from pathlib import Path

        schema_dir = Path(parent_dir) / "application/schemas"
        schema_file = schema_dir / "pdf_page.sd"

        logger.debug(f"Writing schema to {schema_file}")

        schema_file.write_text(settings.schema)

        import time
        time.sleep(2)

        import subprocess

        # Store current directory
        current_dir = os.getcwd()

        try:
            logger.debug("Running deploy commands")

            # Change to application directory
            os.chdir(app_dir)

            process = subprocess.Popen(
                ["vespa", "deploy", "--wait", "500", "-a", f"{VESPA_TENANT_NAME}.{VESPA_APPLICATION_NAME}.{VESPA_INSTANCE_NAME}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            endpoint_url = ""
            for line in iter(process.stdout.readline, ''):
                logger.info(line.strip())
                if "Found endpoints:" in line:
                    # Read the next two lines to get to the URL line
                    next(process.stdout)  # Skip "- dev.aws-us-east-*" line
                    url_line = next(process.stdout)
                    # Extract URL from line like " |-- https://d110fb1d.f78833a9.z.vespa-app.cloud/ (cluster '*_container')"
                    match = re.search(r'https://[^\s]+', url_line)
                    if match:
                        endpoint_url = match.group(0)
                        logger.info(f"Found endpoint URL: {endpoint_url}")

            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, "vespa deploy")

            if not endpoint_url:
                raise Exception("Failed to find endpoint URL in deployment output")

            # Save endpoint_url to the database
            await request.app.db.update_settings(user_id, {"vespa_cloud_endpoint": endpoint_url})

            logger.info(f"Deployment completed successfully! Endpoint URL: {endpoint_url}")
        finally:
            # Always restore the original working directory
            os.chdir(current_dir)

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

def copy_api_key_file(parent_dir, tenant_name):
    storage_dir = os.path.join(parent_dir, "application")
    # Find any .pem file in the given directory
    api_key_src = next((f for f in os.listdir(storage_dir) if f.endswith('.pem')), None)
    if api_key_src:
        api_key_src = os.path.join(storage_dir, api_key_src)
    else:
        raise FileNotFoundError("No .pem file found in application directory")

    api_key_dest = os.path.expanduser("~/.vespa")
    os.makedirs(api_key_dest, exist_ok=True)

    dest_filename = f"{tenant_name}.api-key.pem"
    dest_path = os.path.join(api_key_dest, dest_filename)

    if os.path.exists(api_key_src):
        import shutil
        shutil.copy2(api_key_src, dest_path)
        logger.info(f"Copied API key from {api_key_src} to {dest_path}")
    else:
        logger.warning(f"API key file not found at {api_key_src}")
