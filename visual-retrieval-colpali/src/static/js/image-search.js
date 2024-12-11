function handleImageUpload(input) {
    if (!input.files || !input.files[0]) return;

    // Show modal before starting upload
    const modalContainer = document.createElement('div');
    modalContainer.id = 'image-search-modal';
    document.body.appendChild(modalContainer);

    htmx.ajax('GET', '/image-search-modal', {
        target: '#image-search-modal',
        swap: 'innerHTML'
    });

    const formData = new FormData();
    formData.append('image', input.files[0]);

    fetch('/api/image-search', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        // Remove modal
        const modal = document.getElementById('image-search-modal');
        if (modal) modal.remove();

        // Redirect to search with image query
        const searchParams = new URLSearchParams({
            image_query: data.query_id
        });
        window.location.href = `/search?${searchParams.toString()}`;
    })
    .catch(error => {
        console.error('Error:', error);
        // Remove modal and show error
        const modal = document.getElementById('image-search-modal');
        if (modal) modal.remove();

        alert('Error processing image: ' + error.message);
    });
}
