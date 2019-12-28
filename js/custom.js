$(window).scroll(function () {
    $('header').toggleClass('scrolled', $(this).scrollTop() > 50);
    $('.top-toast').toggleClass('scrolled', $(this).scrollTop() > 50);
    backTop();
});
$(document).ready(function () {
    equalHeight();
    backTop();
    shadowBoxHeight();
    newsTopScroll();
    homeEqualHeight();

    $('button.navbar-toggler').click(function (event) {
        $('header').toggleClass('header-collapse');
    });
    $('header').toggleClass('scrolled', $(this).scrollTop() > 50);
    
    // scroll body to 0px on click
    $('#back-to-top').click(function () {
        $('body,html').animate({
            scrollTop: 0
        }, 400);
        $('#ONNXLogo').focus();
        return false;
    });

    $(document).click(function (event) {
        var clickover = $(event.target);
        var _opened = $(".navbar-collapse").hasClass("show");
        if (_opened === true && !clickover.hasClass("navbar-toggler")) {
            $(".navbar-toggler").click();
        }
    });

    $('#listbox-5').focus(function(){
        var top =  $('.get-started-section').offset().top
        $(window).scrollTop(top);
    });
      
      $(document).keyup(function(e) {
        if($('#navbarNav').hasClass('show')){
            if (e.keyCode === 27) $('button.navbar-toggler').click();   // esc
        }
      });

      $('.btn-getStarted').click(function(){
        var tableTop= $('#getStartedTable').offset().top;
        $('body,html').animate({
            scrollTop: tableTop-100
        }, 600);
      });

      $('.top-toast p > a').click(function(){
          debugger
        var newTop= $(this).attr('href').splite('#');
        $('body,html').animate({
            scrollTop: newTop-100
        }, 600);
      });

      $(document).on("click",".top-toast p > a",function() {
         var firstEle= $(this).attr('href').split('#');
         var newTop = $('#'+firstEle[1]).offset();
        $('body,html').animate({
            scrollTop: newTop.top-120
        }, 400);
    });

      $(document).on('focus',function(e) {
        var docTop = $(this).offset().top;
        $(window).scrollTop(docTop);
      });

      $.get("js/news.json", function(data, status){
        var newsSlide = '';
        newsSlide+='<p class="mb-0 my-2"><span class="bold-text">NEWS: </span>'+data["newsContent"][0].newsTitle;
        newsSlide+='<a aria-label="Read more about news" class="text-white ml-2 bold-text link p-2" href="./news.html#'+data["newsContent"][0].newsId+'"><span class="link-content">Read more</span><span class="link-arrow fa fa-angle-right"></span></a></p>';    
        $('#newsSlider').html(newsSlide);
    });
    
    if (window.innerWidth < 768){
        setTimeout(function(){
            var newsHeight = $('#newsSlider').height();
            $('.content-wrapper').css('padding-top',newsHeight+16);
        },100);
    }
 
    $("ul.navbar-nav li:last-child a").focusout(function(){
        if(window.innerWidth < 992){
            $('button.navbar-toggler').click();            
        } 
      });

}); 


$(window).resize(function(){
    equalHeight();
    shadowBoxHeight();
    homeEqualHeight();
    topNewsHeight();
});

function backTop(){
    if ($(this).scrollTop() > 50) {
        $('#back-to-top').fadeIn();
    } else {
        $('#back-to-top').fadeOut();
    }
}

function newsTopScroll() {
    if(location.hash){
        setTimeout(function(){
            var windowsHash = location.hash;     
            if (windowsHash.length) {
                var tableTop= $(windowsHash).offset().top;
                $(windowsHash).attr('tabindex','-1').focus();
            }
            $('body,html').animate({
                scrollTop: tableTop-150
            }, 800);
        },500);
    }
}

function equalHeight() {
    if (window.innerWidth > 767) {
        var maxHeight = 0;
        $(".equalHeight .bg-lightblue").height('auto');
        $(".equalHeight .bg-lightblue").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight .bg-lightblue").height(maxHeight);

        var maxHeight = 0;
        $(".content-equalHeight .bg-lightblue p").height('auto');
        $(".content-equalHeight .bg-lightblue p").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".content-equalHeight .bg-lightblue p").height(maxHeight);

        var maxHeight = 0;
        $(".content-equalHeight .bg-lightblue h3").height('auto');
        $(".content-equalHeight .bg-lightblue h3").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".content-equalHeight .bg-lightblue h3").height(maxHeight);
        
        var maxHeight = 0;
        $(".equalHeight-1 .bg-lightblue").height('auto');
        $(".equalHeight-1 .bg-lightblue").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight-1 .bg-lightblue").height(maxHeight);

        var maxHeight = 0;
        $(".equalHeight-1 .onnx-model-content p").height('auto');
        $(".equalHeight-1 .onnx-model-content p").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight-1 .onnx-model-content p").height(maxHeight);

        
        var maxHeight = 0;
        $(".equalHeight .build-model").height('auto');
        $(".equalHeight .build-model").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight .build-model").height(maxHeight);

        var maxHeight = 0;
        $(".equalHeight-1 .additional-tools").height('auto');
        $(".equalHeight-1 .additional-tools").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight-1 .additional-tools").height(maxHeight);

        var maxHeight = 0;
        $(".equalHeight-1 .additional-tools p").height('auto');
        $(".equalHeight-1 .additional-tools p").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight-1 .additional-tools p").height(maxHeight);

        var maxHeight = 0;
        $(".equalHeight .build-model p").height('auto');
        $(".equalHeight .build-model p").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight .build-model p").height(maxHeight);
      
    } else {
        $(".equalHeight .bg-lightblue").height('auto');
        $(".equalHeight-1 .bg-lightblue").height('auto');
        $(".equalHeight-1 .onnx-model-content p").height('auto');
        $(".equalHeight .build-model").height('auto');
        $(".equalHeight-1 .additional-tools").height('auto');
        $(".equalHeight-1 .additional-tools p").height('auto');
        $(".equalHeight .build-model p").height('auto');
        $(".content-equalHeight .bg-lightblue p").height('auto');
        $(".content-equalHeight .bg-lightblue h3").height('auto');
    }
}

function homeEqualHeight(){
    if (window.innerWidth > 767) {
        var maxHeight = 0;
        $(".equalHeight-2 .col-news h3").height('auto');
        $(".equalHeight-2 .col-news h3").each(function () {
            if ($(this).height() > maxHeight) {
                maxHeight = $(this).height();
            }
        });
        $(".equalHeight-2 .col-news h3").height(maxHeight);
        
        setTimeout(function(){
            var maxHeight = 0;
            $(".equalHeight-2 .col-news p").height('auto');
            $(".equalHeight-2 .col-news p").each(function () {
                if ($(this).height() > maxHeight) {
                    maxHeight = $(this).height();
                }
            });
            $(".equalHeight-2 .col-news p").height(maxHeight);  
        },400);
      
    }else {
        $(".equalHeight-2 .col-news h3").height('auto');
        $(".equalHeight-2 .col-news p").height('auto');
    }
    
}

function shadowBoxHeight(){
    if (window.innerWidth < 767){
        let boxHeight=$('.shadow-box .shadow').height();
        $('.page-content .key-benefits-wrapper').css('padding-top',boxHeight-65);
    }else{
        $('.page-content .key-benefits-wrapper').css('padding-top','initial');
    }
}

function topNewsHeight(){
    var newsHeight = $('#newsSlider').height();
    $('.content-wrapper').css('padding-top',newsHeight+16);
}
