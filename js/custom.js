fetch("js/news.json")
    .then(r => r.json())
    .then(data => {
        const item = data["newsContent"][0];
        const slider = document.getElementById('newsSlider');
        if (slider) {
            slider.innerHTML =
                '<p class="mb-0 my-2"><span class="bold-text">NEWS: </span>' + item.newsTitle +
                '<a aria-label="Read more about news" class="text-white ms-1 bold-text link p-2" href="./news.html#' + item.newsId +
                '"><span class="link-content">Read more</span><span class="link-arrow fa fa-angle-right"></span></a></p>';
        }
    });

window.addEventListener('scroll', function () {
    const scrolled = window.scrollY > 50;
    document.querySelector('header').classList.toggle('scrolled', scrolled);
    const toast = document.querySelector('.top-toast');
    if (toast) toast.classList.toggle('scrolled', scrolled);
    backTop();
});

document.addEventListener('DOMContentLoaded', function () {
    equalHeight();
    backTop();
    shadowBoxHeight();
    newsTopScroll();

    setTimeout(function () {
        homeEqualHeight();
        topNewsInnerHeight();
        topNewsHeight();
    }, 700);

    const navItem = window.location.pathname.split('/').pop().split('.')[0];
    setTimeout(function () {
        document.querySelectorAll('.navbar-custom .active').forEach(el => el.classList.remove('active'));
        const activeEl = document.querySelector('.navbar-custom .' + navItem);
        if (activeEl) activeEl.classList.add('active');
    }, 300);

    const toggler = document.querySelector('button.navbar-toggler');
    if (toggler) {
        toggler.addEventListener('click', function () {
            document.querySelector('header').classList.toggle('header-collapse');
        });
    }

    document.querySelector('header').classList.toggle('scrolled', window.scrollY > 50);

    const backToTopBtn = document.getElementById('back-to-top');
    if (backToTopBtn) {
        backToTopBtn.addEventListener('click', function (e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
            document.getElementById('ONNXLogo').focus();
        });
    }

    document.addEventListener('click', function (event) {
        const navbarCollapse = document.querySelector('.navbar-collapse');
        const isOpen = navbarCollapse && navbarCollapse.classList.contains('show');
        if (isOpen && !event.target.closest('.navbar-toggler')) {
            document.querySelector('.navbar-toggler').click();
        }
    });

    const listbox = document.getElementById('listbox-5');
    if (listbox) {
        listbox.addEventListener('focus', function () {
            const section = document.querySelector('.get-started-section');
            if (section) window.scrollTo({ top: section.offsetTop, behavior: 'smooth' });
        });
    }

    document.addEventListener('keyup', function (e) {
        const navbarNav = document.getElementById('navbarNav');
        if (navbarNav && navbarNav.classList.contains('show') && e.key === 'Escape') {
            document.querySelector('button.navbar-toggler').click();
        }
    });

    document.querySelectorAll('.btn-getStarted').forEach(function (btn) {
        btn.addEventListener('click', function () {
            const table = document.getElementById('getStartedTable');
            if (table) window.scrollTo({ top: table.offsetTop - 100, behavior: 'smooth' });
        });
    });

    document.addEventListener('click', function (e) {
        const link = e.target.closest('.top-toast p > a');
        if (link) {
            e.preventDefault();
            const id = link.getAttribute('href').split('#')[1];
            const target = document.getElementById(id);
            if (target) window.scrollTo({ top: target.offsetTop - 120, behavior: 'smooth' });
        }
    });

    const lastNavLink = document.querySelector('ul.navbar-nav li:last-child a');
    if (lastNavLink) {
        lastNavLink.addEventListener('focusout', function () {
            if (window.innerWidth < 992) {
                document.querySelector('button.navbar-toggler').click();
            }
        });
    }

    if (window.innerWidth < 768) {
        setTimeout(function () {
            const slider = document.getElementById('newsSlider');
            const wrapper = document.querySelector('.content-wrapper');
            if (slider && wrapper) wrapper.style.paddingTop = (slider.offsetHeight + 16) + 'px';
        }, 100);
    }
});

window.addEventListener('resize', function () {
    equalHeight();
    shadowBoxHeight();
    homeEqualHeight();
    topNewsHeight();
    topNewsInnerHeight();
});

function backTop() {
    const btn = document.getElementById('back-to-top');
    if (!btn) return;
    btn.style.display = window.scrollY > 50 ? 'block' : 'none';
}

function newsTopScroll() {
    if (!location.hash) return;
    setTimeout(function () {
        const target = document.querySelector(location.hash);
        if (target) {
            target.setAttribute('tabindex', '-1');
            target.focus();
            window.scrollTo({ top: target.offsetTop - 150, behavior: 'smooth' });
        }
    }, 500);
}

function setMaxHeight(selector) {
    const els = document.querySelectorAll(selector);
    els.forEach(el => el.style.height = 'auto');
    if (window.innerWidth > 767) {
        let max = 0;
        els.forEach(el => { if (el.offsetHeight > max) max = el.offsetHeight; });
        els.forEach(el => el.style.height = max + 'px');
    }
}

function equalHeight() {
    setMaxHeight('.equalHeight .bg-lightblue');
    setMaxHeight('.content-equalHeight .bg-lightblue p');
    setMaxHeight('.content-equalHeight .bg-lightblue h3');
    setMaxHeight('.equalHeight-1 .bg-lightblue');
    setMaxHeight('.equalHeight-1 .onnx-model-content p');
    setMaxHeight('.equalHeight .build-model');
    setMaxHeight('.equalHeight-1 .additional-tools');
    setMaxHeight('.equalHeight-1 .additional-tools p');
    setMaxHeight('.equalHeight .build-model p');
}

function homeEqualHeight() {
    setMaxHeight('.equalHeight-2 .col-news h3');
    setMaxHeight('.equalHeight-2 .col-news p');
}

function shadowBoxHeight() {
    const wrapper = document.querySelector('.page-content .key-benefits-wrapper');
    if (!wrapper) return;
    if (window.innerWidth < 767) {
        const box = document.querySelector('.shadow-box .shadow');
        if (box) wrapper.style.paddingTop = (box.offsetHeight - 65) + 'px';
    } else {
        wrapper.style.paddingTop = 'initial';
    }
}

function topNewsHeight() {
    const slider = document.getElementById('newsSlider');
    const wrapper = document.querySelector('.content-wrapper');
    if (slider && wrapper) wrapper.style.paddingTop = (slider.offsetHeight + 16) + 'px';
}

function topNewsInnerHeight() {
    const slider = document.getElementById('newsSlider');
    const banner = document.querySelector('.innerpage-main-wrapper .main-wrapper .top-banner-bg');
    if (slider && banner) {
        const padding = window.innerWidth < 991 ? slider.offsetHeight + 128 : slider.offsetHeight + 172;
        banner.style.paddingTop = padding + 'px';
    }
}
