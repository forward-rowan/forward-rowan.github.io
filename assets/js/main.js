// 平滑滚动效果
document.addEventListener('DOMContentLoaded', function() {
    // 添加鼠标跟随效果
    const shapes = document.querySelectorAll('.shape');
    
    document.addEventListener('mousemove', function(e) {
        const mouseX = e.clientX;
        const mouseY = e.clientY;
        
        shapes.forEach((shape, index) => {
            const speed = (index + 1) * 0.02;
            const x = (mouseX - window.innerWidth / 2) * speed;
            const y = (mouseY - window.innerHeight / 2) * speed;
            
            shape.style.transform = `translate(${x}px, ${y}px)`;
        });
    });
    
    // 添加卡片点击动画
    const postCards = document.querySelectorAll('.post-card');
    postCards.forEach(card => {
        card.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
});
